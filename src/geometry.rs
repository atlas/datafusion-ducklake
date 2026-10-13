//! Geometry columns, stored the way DuckDB's DuckLake extension stores them.
//!
//! A geometry column's catalog type is `geometry`. Its values are WKB in a Parquet
//! `BYTE_ARRAY` annotated with Parquet's `GEOMETRY` logical type, which the `parquet`
//! crate's `geospatial` feature maps to and from Arrow's `geoarrow.wkb` extension type.
//! DuckDB keeps no coordinate reference system in the lake: the catalog type is plain
//! `geometry` and the Parquet type carries none, so neither is written here and a CRS
//! on an incoming field is dropped. Callers that need one keep it alongside the lake.
//!
//! A geometry column's statistics leave `min_value` and `max_value` `NULL` and put a
//! bounding box and the shapes present in `extra_stats`, as
//! `{"bbox": {"xmin": …, …, "mmax": …}, "types": ["point", "polygon_z", …]}`.
//!
//! Only top-level columns are geometry here. A geometry nested in a struct, list or map
//! is refused on write and read back as WKB bytes without the extension type.

use arrow::datatypes::{DataType, Field};
use parquet::geospatial::statistics::GeospatialStatistics;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::Result;
use crate::error::DuckLakeError;

/// The catalog type of a geometry column.
pub const GEOMETRY_TYPE: &str = "geometry";

/// Arrow's extension type for WKB geometry.
pub const WKB_EXTENSION: &str = "geoarrow.wkb";

const EXTENSION_NAME_KEY: &str = "ARROW:extension:name";
const EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";

/// Whether `ducklake_type` names geometry: `geometry`, or one of the shape names
/// older DuckLake catalogs used.
pub fn is_geometry_type(ducklake_type: &str) -> bool {
    matches!(
        ducklake_type.trim().to_ascii_lowercase().as_str(),
        "geometry"
            | "point"
            | "linestring"
            | "polygon"
            | "multipoint"
            | "multilinestring"
            | "multipolygon"
            | "geometrycollection"
            | "linestring z"
    )
}

/// The Arrow field a top-level geometry column reads as: WKB marked `geoarrow.wkb`,
/// with the empty metadata a Parquet `GEOMETRY` column without a CRS reads back with.
pub fn geometry_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Binary, nullable).with_metadata(
        [
            (EXTENSION_NAME_KEY.to_string(), WKB_EXTENSION.to_string()),
            (EXTENSION_METADATA_KEY.to_string(), "{}".to_string()),
        ]
        .into(),
    )
}

/// Whether `field` holds WKB geometry. Any other GeoArrow encoding is refused: its
/// layout would otherwise be stored as an ordinary struct or list column.
pub fn is_geometry_field(field: &Field) -> Result<bool> {
    match field.metadata().get(EXTENSION_NAME_KEY).map(String::as_str) {
        Some(WKB_EXTENSION) => match field.data_type() {
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Ok(true),
            other => Err(DuckLakeError::UnsupportedType(format!(
                "column '{}' is {WKB_EXTENSION} but holds {other}, not binary",
                field.name()
            ))),
        },
        Some(name) if name.starts_with("geoarrow.") => {
            Err(DuckLakeError::UnsupportedType(format!(
                "column '{}' is {name}; encode geometry as {WKB_EXTENSION} to store it",
                field.name()
            )))
        },
        _ => Ok(false),
    }
}

/// Refuses a geometry anywhere inside `data_type`, which DuckLake would need nested
/// geometry support to store as such.
pub(crate) fn refuse_nested_geometry(column: &str, data_type: &DataType) -> Result<()> {
    let children: Vec<&Field> = match data_type {
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => vec![f],
        DataType::Map(f, _) => vec![f],
        DataType::Struct(fields) => fields.iter().map(AsRef::as_ref).collect(),
        _ => return Ok(()),
    };
    for child in children {
        if is_geometry_field(child)? {
            return Err(DuckLakeError::UnsupportedType(format!(
                "column '{column}' nests geometry field '{}'; only top-level geometry columns are supported",
                child.name()
            )));
        }
        refuse_nested_geometry(column, child.data_type())?;
    }
    Ok(())
}

/// A geometry column's bounding box and shapes, as `extra_stats` holds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryStats {
    pub bbox: BoundingBox,
    /// Shape names such as `point` or `multipolygon_zm`, sorted.
    pub types: BTreeSet<String>,
}

/// An axis-aligned box. Z and M are `None` when no value had them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub zmin: Option<f64>,
    pub zmax: Option<f64>,
    pub mmin: Option<f64>,
    pub mmax: Option<f64>,
}

impl GeometryStats {
    /// The stats of one Parquet column chunk. `None` when the chunk recorded no box,
    /// no shapes, or a box that wraps around the antimeridian.
    pub fn from_parquet(stats: &GeospatialStatistics) -> Option<Self> {
        let bbox = stats.bounding_box()?;
        if !(bbox.get_xmin() <= bbox.get_xmax() && bbox.get_ymin() <= bbox.get_ymax()) {
            return None;
        }
        let types = stats
            .geospatial_types()?
            .iter()
            .map(|code| shape_name(*code))
            .collect::<Option<BTreeSet<_>>>()?;
        let range = |valid: bool, min: Option<f64>, max: Option<f64>| match (valid, min, max) {
            (true, Some(min), Some(max)) if min <= max => (Some(min), Some(max)),
            _ => (None, None),
        };
        let (zmin, zmax) = range(bbox.is_z_valid(), bbox.get_zmin(), bbox.get_zmax());
        let (mmin, mmax) = range(bbox.is_m_valid(), bbox.get_mmin(), bbox.get_mmax());
        Some(Self {
            bbox: BoundingBox {
                xmin: bbox.get_xmin(),
                xmax: bbox.get_xmax(),
                ymin: bbox.get_ymin(),
                ymax: bbox.get_ymax(),
                zmin,
                zmax,
                mmin,
                mmax,
            },
            types,
        })
    }

    /// Reads `extra_stats`. `None` for anything else a writer put there.
    pub fn parse(extra_stats: &str) -> Option<Self> {
        serde_json::from_str(extra_stats).ok()
    }

    /// The `extra_stats` text.
    pub fn encode(&self) -> String {
        serde_json::to_string(self).expect("geometry stats serialize")
    }

    /// The smallest stats covering both.
    pub fn merge(&self, other: &Self) -> Self {
        let b = |a: Option<f64>, c: Option<f64>, f: fn(f64, f64) -> f64| match (a, c) {
            (Some(a), Some(c)) => Some(f(a, c)),
            (a, c) => a.or(c),
        };
        Self {
            bbox: BoundingBox {
                xmin: self.bbox.xmin.min(other.bbox.xmin),
                xmax: self.bbox.xmax.max(other.bbox.xmax),
                ymin: self.bbox.ymin.min(other.bbox.ymin),
                ymax: self.bbox.ymax.max(other.bbox.ymax),
                zmin: b(self.bbox.zmin, other.bbox.zmin, f64::min),
                zmax: b(self.bbox.zmax, other.bbox.zmax, f64::max),
                mmin: b(self.bbox.mmin, other.bbox.mmin, f64::min),
                mmax: b(self.bbox.mmax, other.bbox.mmax, f64::max),
            },
            types: self.types.union(&other.types).cloned().collect(),
        }
    }

    /// Merges several chunks' or files' stats. `None` if any is unknown, because the
    /// merged box would then not cover every value.
    pub fn merge_all<'a>(stats: impl IntoIterator<Item = Option<&'a Self>>) -> Option<Self> {
        let mut merged: Option<Self> = None;
        for stats in stats {
            let stats = stats?;
            merged = Some(match merged {
                Some(merged) => merged.merge(stats),
                None => stats.clone(),
            });
        }
        merged
    }
}

/// DuckDB's name for an ISO WKB geometry type code: `point`, `linestring_z`, …
fn shape_name(code: i32) -> Option<String> {
    let base = match code % 1000 {
        1 => "point",
        2 => "linestring",
        3 => "polygon",
        4 => "multipoint",
        5 => "multilinestring",
        6 => "multipolygon",
        7 => "geometrycollection",
        _ => return None,
    };
    let suffix = match code / 1000 {
        0 => "",
        1 => "_z",
        2 => "_m",
        3 => "_zm",
        _ => return None,
    };
    Some(format!("{base}{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::geospatial::bounding_box::BoundingBox as ParquetBox;

    #[test]
    fn reads_what_duckdb_writes() {
        let duckdb = r#"{"bbox": {"xmin": 0.000000, "xmax": 1.000000, "ymin": 0.000000, "ymax": 2.000000, "zmin": 0.000000, "zmax": 3.000000, "mmin": null, "mmax": null}, "types": ["linestring_m", "point"]}"#;
        let stats = GeometryStats::parse(duckdb).unwrap();
        assert_eq!(stats.bbox.ymax, 2.0);
        assert_eq!(stats.bbox.zmax, Some(3.0));
        assert_eq!(stats.bbox.mmin, None);
        assert_eq!(GeometryStats::parse(&stats.encode()), Some(stats));
    }

    #[test]
    fn shape_names_follow_duckdb() {
        let stats = GeospatialStatistics::new(
            Some(ParquetBox::new(0.0, 1.0, 0.0, 1.0)),
            Some(vec![1001, 2002, 3006, 3]),
        );
        let types = GeometryStats::from_parquet(&stats).unwrap().types;
        assert_eq!(
            types.into_iter().collect::<Vec<_>>(),
            ["linestring_m", "multipolygon_zm", "point_z", "polygon"]
        );
    }

    #[test]
    fn an_unknown_part_makes_the_merge_unknown() {
        let a = GeometryStats::from_parquet(&GeospatialStatistics::new(
            Some(ParquetBox::new(0.0, 1.0, 0.0, 1.0)),
            Some(vec![1]),
        ))
        .unwrap();
        let b = GeometryStats::from_parquet(&GeospatialStatistics::new(
            Some(ParquetBox::new(-5.0, 0.5, 2.0, 3.0).with_zrange(1.0, 4.0)),
            Some(vec![1003]),
        ))
        .unwrap();
        let merged = GeometryStats::merge_all([Some(&a), Some(&b)]).unwrap();
        assert_eq!(
            (
                merged.bbox.xmin,
                merged.bbox.xmax,
                merged.bbox.ymin,
                merged.bbox.ymax
            ),
            (-5.0, 1.0, 0.0, 3.0)
        );
        assert_eq!(merged.bbox.zmax, Some(4.0));
        assert_eq!(GeometryStats::merge_all([Some(&a), None]), None);
    }

    #[test]
    fn other_geoarrow_encodings_are_refused() {
        let point = Field::new("g", DataType::Binary, true)
            .with_metadata([(EXTENSION_NAME_KEY.to_string(), "geoarrow.point".to_string())].into());
        assert!(is_geometry_field(&point).is_err());
        assert!(is_geometry_field(&geometry_field("g", true)).unwrap());
        assert!(!is_geometry_field(&Field::new("g", DataType::Binary, true)).unwrap());
    }
}
