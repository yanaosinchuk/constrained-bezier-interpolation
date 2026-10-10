# Example data

The numerical experiment uses the processed time-series file:

`pegelonline_leunneu_2024.xlsx`

The workbook contains the two columns expected by the comparison script:

```text
timestamp
value
```

The included experiment contains 35,136 reference observations and uses a support-point spacing of 672 samples.

## Source

The underlying hydrological data are from **PEGELONLINE**, provided by the
German Federal Waterways and Shipping Administration (WSV), for the gauge
**LEUN NEU** on the Lahn (station number **25800200**).

Official station metadata:

https://www.pegelonline.wsv.de/gast/stammdaten?pegelnr=25800200

PEGELONLINE describes its downloadable historical measurements as unverified
raw hydrological data. These data are made available under the
**Data Licence Germany – Zero – Version 2.0 (DL-DE Zero-2.0)**.

PEGELONLINE data/download information:

https://pegelonline.wsv.de/webservice/downloads

Licence text:

https://www.govdata.de/dl-de/zero-2-0

The Excel file in this repository is a processed project copy prepared for the
interpolation experiment; the original public-data licence applies to the
underlying PEGELONLINE measurements.
