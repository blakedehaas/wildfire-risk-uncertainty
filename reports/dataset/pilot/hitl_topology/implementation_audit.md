# Checkpoint 4A source topology implementation audit

The independent diagnostic uses `STRtree` over projected source geometries for every BP-valid pilot cell. It sorts candidate indices by OBJECTID and intersects each source polygon with the cell once. It does not use the constructor’s affine candidate-range loop or its WUI aggregation results.

## Retrieval and accumulation

| Pilot window | BP-valid cells | retrieved rows | valid polygons | invalid polygons | duplicate retrieved OBJECTIDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| row_00999_col_01491_h008_w008 | 64 | 147 | 147 | 0 | 0 |
| row_03411_col_00158_h008_w008 | 7 | 1 | 1 | 0 | 0 |
| row_04215_col_11319_h008_w008 | 64 | 2 | 2 | 0 | 0 |
| row_04218_col_11297_h008_w008 | 64 | 528 | 528 | 0 | 0 |
| row_05056_col_05894_h064_w064 | 4096 | 12410 | 12405 | 5 | 0 |
| row_05717_col_07786_h008_w008 | 64 | 9 | 9 | 0 | 0 |

- Duplicate retrieved OBJECTIDs: 0.
- Duplicate spatial-tree indices: 0.
- Duplicate OBJECTID/cell contributions: 0.
- Valid MultiPolygons checked: 13092; part-area mismatches: 0.
- Reprojection part-count/type changes: 0.
- Duplicate BP-valid cell IDs across tiles: 0 (asserted).
- Independent raw coverage mismatches against canonical pilot: 0.
- Independent overlap and excess cell sets match canonical diagnostics exactly: 434 and 254 cells.

The source query returns distinct OBJECTIDs within every tile. The canonical constructor sorts polygon candidates by OBJECTID and iterates each native cell coordinate once per polygon. Shapely projects each MultiPolygon as one geometry; `intersection(cell)` is called once per OBJECTID/cell. The independent spatial tree yields the same raw coverage, overlap cells and excess cells. The fixed pilot windows have disjoint cell IDs. These checks show no duplicate contribution from retrieval, part handling, spatial ordering or tile boundaries.

## Interpretation limits

Of 660 projected positive-area pairs, 435 also have a positive source-CRS intersection projected into that cell; 144 have source-CRS overlap elsewhere but none found in that cell; and 81 have zero source-CRS pair-intersection area. The latter two groups indicate projection/overlay representation effects in the cell, not duplicate retrieval.
For the 254 excess-coverage cells, total excess is 0.56222519814 m², duplicate coverage is 0.816875711113 m², and uncovered gap is 0.254650512972 m². The identity excess = duplicate coverage − uncovered gap holds within floating-point error.

Positive overlaps are intersections of distinct, individually valid projected source geometries. The five invalid source geometries are skipped by the canonical overlay, so they do not directly cause those overlap pairs. Their bounding envelopes mask affected cells separately. The overlap areas and source attributes are in the CSV tables. Tiny positive areas may reflect source coordinate precision, but choosing a tolerance or topology resolution is a scientific decision and is not made here.
The `make_valid` metrics are diagnostic only. A repaired geometry has not been used in canonical output.
