# Integrity Check

In our paper [valentin2025predictive](@cite) we introduce an "integrity check" that let's us compare the reprojection error of the estimated pose to the magnitude of the predicted uncertainties by adoping an algorithm similar to RAIM [joerger2014solution](@cite).
This figure provides a brief illustration, with more details in [valentin2025predictive](@cite).

![An illustration of the integrity check.](figs/raim_explanation.svg)

```@docs; canonical = false
compute_integrity_statistic
```

```@bibliography
Pages = ["integrity_check.md"]
Canonical = true
```
