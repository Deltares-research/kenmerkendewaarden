# What's new

## 0.3.0 (2024-10-01)

### Feat
- from meters to centimeters and other updates for dashboard in [#142](https://github.com/Deltares-research/kenmerkendewaarden/pull/142)
- added tidalrange to slotgemiddelden and outputformats to accomodate SAS dashboard in [#142](https://github.com/Deltares-research/kenmerkendewaarden/pull/142)

### Fix
- accomodate updated astrog datetime handling in hatyan 2.9.0 in [#141](https://github.com/Deltares-research/kenmerkendewaarden/pull/141)
- more stable coverage computation in `calc_HWLWtidalindicators()` and `calc_havengetallen()` in [#146](https://github.com/Deltares-research/kenmerkendewaarden/pull/146)


## 0.2.0 (2024-09-02)
This version has a reduced complexity for the user by making the public functions straightforward to use. There will be no changes that significantly affect the results, so we can still (approximately) reproduce the kenmerkendewaarden 2011/2021 derived in 2020-2022. The resulting methods are still not production-ready, since there are still improvements needed that will be inplemented in kenmerkendewaarden 0.3.0.

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
- added dedicated plotting functions in [#64](https://github.com/Deltares-research/kenmerkendewaarden/pull/64), [#66](https://github.com/Deltares-research/kenmerkendewaarden/pull/66) and [#68](https://github.com/Deltares-research/kenmerkendewaarden/pull/68)
- added computation of hat/lat from measurements with `kw.calc_hat_lat_frommeasurements()` in [#74](https://github.com/Deltares-research/kenmerkendewaarden/pull/74)
- added modular check for timeseries coverage in [#76](https://github.com/Deltares-research/kenmerkendewaarden/pull/76)
- simplified methods for overschrijdingsfreqs and reducing public functions to `kw.calc_overschrijding()` in [#81](https://github.com/Deltares-research/kenmerkendewaarden/pull/81)
- add station attribute to measurements in [#96](https://github.com/Deltares-research/kenmerkendewaarden/pull/96) and [#108](https://github.com/Deltares-research/kenmerkendewaarden/pull/108)
- add hydra statistics for overschrijdingen as preparation for eventual method in [#106](https://github.com/Deltares-research/kenmerkendewaarden/pull/106)
- improvements to output csv files in [#109](https://github.com/Deltares-research/kenmerkendewaarden/pull/109)
- drop duplicate times in `kw.read_measurements()` in [#116](https://github.com/Deltares-research/kenmerkendewaarden/pull/116)
- use NETCDF4_CLASSIC format to reduce file sizes written by `kw.retrieve_measurements()` in [#119](https://github.com/Deltares-research/kenmerkendewaarden/pull/119)
- align output of all methods in [#127](https://github.com/Deltares-research/kenmerkendewaarden/pull/127)

### Fix
- implemented workaround for pandas 2.2.0 with different rounding behaviour in [#69](https://github.com/Deltares-research/kenmerkendewaarden/pull/69)
- fixed different lengths of `compute_expected_counts()` and `compute_actual_counts()` in case of all-nan periods in [#87](https://github.com/Deltares-research/kenmerkendewaarden/pull/87)
- clearer error message in case of too many nans in timeseries slotgemiddelde model fit in [#89](https://github.com/Deltares-research/kenmerkendewaarden/pull/89)


## 0.1.0 (2024-03-11)
This is the set of kenmerkende waarden kust scripts and functions as transfered from hatyan and how they were applied in the kwk-2022 project.

### Feat
- added scripts with functions and examples from hatyan repository in [#3](https://github.com/Deltares-research/kenmerkendewaarden/pull/3)
- updated scripts to work with hatyan==2.7.0 which still contained ddl and kwk functions in [#7](https://github.com/Deltares-research/kenmerkendewaarden/pull/7)
