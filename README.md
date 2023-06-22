# RoRaP (**Ro**tational **Ra**man Screening of **P**lasma Systems)
The documentation to RoRaP will be made available via a program website.

## Quick ReadMe
The program is responsible for a simpler reading-out of rotational Raman spectra collected via screening a plasma system. 

`__init__.py` is an arbitrary code file responsible for loading the library. The actual loading of an experimental spectral image is accomplished by the `fiber.py` module.
`filters.py` introduces a simpler Gaussian filter further used in estimating the noise level `noise.py` and, subsequently, the spectral base line (`bline.py`). Experimental signal peaks may be detected by using `pdetect.py` and assigned via `assign.py`. The peaks detected may then undergo deterministic or stochastic numerical fitting of line or band profiles (`pfit.py`). Finally, `raman.py` contains computational tools for drawing a theoretical rotational spectrum based on Dunham series coefficients and selection rules, which may be optimised onto an experimental record to draw temperature and chemical composition data. Limited case studies are suggested by inserting molecular constant dictionaries.

The code was drafted by Vojta Laitl, as partially reproduced from program scripts by Petr Kubelík, Ph.D., and Vojta Laitl at the Czech Academy of Sciences in 2022. The latter is found at [the code script's webpage](https://raw.githack.com/laitvo/PuPlasAn/main/docs/build/html/index.html) and distributed under [Apache License 2.0](https://github.com/laitvo/PuPlasAn/blob/main/LICENSE). All details to particular functions and program modules are to be found in the above code scripts and in the example section below.

## Dependencies
Dependencies are to be constructed.

## Example section
Example section is to be constructed.
