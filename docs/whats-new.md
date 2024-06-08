# What's new

## UNRELEASED

### Feat
- moved from pickle to netcdf and improved statistics in [#14](https://github.com/Deltares-research/kenmerkendewaarden/pull/14)
- structured data retrieval and statistics in [#16](https://github.com/Deltares-research/kenmerkendewaarden/pull/16) and [#19](https://github.com/Deltares-research/kenmerkendewaarden/pull/19) and [#33](https://github.com/Deltares-research/kenmerkendewaarden/pull/33)
- update dependencies and code to hatyan 2.8.0 in [#28](https://github.com/Deltares-research/kenmerkendewaarden/pull/28)
- added neaptide tidal indicators for extremes in [#34](https://github.com/Deltares-research/kenmerkendewaarden/pull/34)
- used threshold frequency instead of fixed index in `kw.overschrijding.blend_distributions` in [#38](https://github.com/Deltares-research/kenmerkendewaarden/pull/38)
- dropped timezones consistently in `kw.calc_wltidalindicators()` and `kw.calc_HWLWtidalindicators()` to increase performance [#41](https://github.com/Deltares-research/kenmerkendewaarden/pull/41)
- simplified methods for gemiddeld getij and reducing public functions to `kw.gemiddeld_getijkromme_av_sp_np()` in [#46](https://github.com/Deltares-research/kenmerkendewaarden/pull/46)


## 0.1.0 (2024-03-11)
This is the set of kenmerkende waarden kust scripts and functions as transfered from hatyan and how they were applied in the kwk-2022 project.

### Feat
- added scripts with functions and examples from hatyan repository in [#3](https://github.com/Deltares-research/kenmerkendewaarden/pull/3)
- updated scripts to work with hatyan==2.7.0 which still contained ddl and kwk functions in [#7](https://github.com/Deltares-research/kenmerkendewaarden/pull/7)
