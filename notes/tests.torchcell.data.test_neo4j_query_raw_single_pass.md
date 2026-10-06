---
id: cht9gk32mi9ac9hwm5js5xr
title: Test_neo4j_query_raw_single_pass
desc: ''
updated: 1790983329919
created: 1790983329919
---

## 2026.10.05 - Exact declared-class map (#640)

The cache fix itself landed in 8533adeb5 (`environment_class_for`, `ENVIRONMENT_CLASSES`). `test_cached_environment_is_safe_to_pass_unvalidated` additionally pins the exact map: `CultureEnvironment` for `strain_environment_response`, `Environment` for every other family, and `ENVIRONMENT_CLASSES` holds exactly those two classes, so a new family or environment subclass has to be added to the test deliberately.
