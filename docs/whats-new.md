# What's new

## UNRELEASED

### Feat
- moved from pickle to netcdf and improved statistics in [#14](https://github.com/Deltares-research/kenmerkendewaarden/pull/14)
- structured data retrieval and statistics in [#16](https://github.com/Deltares-research/kenmerkendewaarden/pull/16) and [#19](https://github.com/Deltares-research/kenmerkendewaarden/pull/19) and [#33](https://github.com/Deltares-research/kenmerkendewaarden/pull/33)
- update dependencies and code to hatyan 2.8.0 in [#28](https://github.com/Deltares-research/kenmerkendewaarden/pull/28)
- added neaptide tidal indicators for extremes in [#34](https://github.com/Deltares-research/kenmerkendewaarden/pull/34)
- used threshold frequency instead of fixed index in `kw.overschrijding.blend_distributions` in [#38](https://github.com/Deltares-research/kenmerkendewaarden/pull/38)
- dropped timezones consistently in `kw.calc_wltidalindicators()` and `kw.calc_HWLWtidalindicators()` to increase performance [#41](https://github.com/Deltares-research/kenmerkendewaarden/pull/41)
- simplified methods for gemiddeld getij and reducing public functions to `kw.calc_gemiddeldgetij()` in [#46](https://github.com/Deltares-research/kenmerkendewaarden/pull/46)
- simplified methods for havengetallen and reducing public functions to `kw.calc_havengetallen()` in [#48](https://github.com/Deltares-research/kenmerkendewaarden/pull/48)
- simplified methods for slotgemiddelden and reducing public functions to `kw.calc_slotgemiddelden()` in [#62](https://github.com/Deltares-research/kenmerkendewaarden/pull/62)
- increased test coverage in [#50](https://github.com/Deltares-research/kenmerkendewaarden/pull/50) and [#55](https://github.com/Deltares-research/kenmerkendewaarden/pull/55)
- created `kw.plot_stations()` in [#57](https://github.com/Deltares-research/kenmerkendewaarden/pull/57)
- clipping of timeseries on physical breaks with `kw.data_retrieve.clip_timeseries_physical_break()` (private) in [#61](https://github.com/Deltares-research/kenmerkendewaarden/pull/61) and [#64](https://github.com/Deltares-research/kenmerkendewaarden/pull/64)


## 0.1.0 (2024-03-11)
This is the set of kenmerkende waarden kust scripts and functions as transfered from hatyan and how they were applied in the kwk-2022 project.

### Feat
- added scripts with functions and examples from hatyan repository in [#3](https://github.com/Deltares-research/kenmerkendewaarden/pull/3)
- updated scripts to work with hatyan==2.7.0 which still contained ddl and kwk functions in [#7](https://github.com/Deltares-research/kenmerkendewaarden/pull/7)
