# [RoRaP](https://raw.githack.com/laitvo/RoRaP/main/docs/html/index.html) (**Ro**tational **Ra**man Screening of **P**lasma Systems)
The documentation to RoRaP is made available via a [program website](https://raw.githack.com/laitvo/RoRaP/main/docs/html/index.html).

## Quick ReadMe
The program is responsible for a simpler reading-out of rotational Raman spectra collected via screening a plasma system. 

`__init__.py` is an arbitrary code file responsible for loading the library. The actual loading of an experimental spectral image is accomplished by the `fiber.py` module.
`filters.py` introduces a simpler Gaussian filter further used in estimating the noise level `noise.py` and, subsequently, the spectral base line (`bline.py`). Experimental signal peaks may be detected by using `pdetect.py` and assigned via `assign.py`. The peaks detected may then undergo deterministic or stochastic numerical fitting of line or band profiles (`pfit.py`). Finally, `raman.py` contains computational tools for drawing a theoretical rotational spectrum based on Dunham series coefficients and selection rules, which may be optimised onto an experimental record to draw temperature and chemical composition data. Limited case studies are suggested by inserting molecular constant dictionaries.

The code was drafted by Vojta Laitl, as partially reproduced from program scripts by Petr Kubelík, Ph.D., at the Czech Academy of Sciences in 2022. The program can be found at [the code script's webpage](https://raw.githack.com/laitvo/PuPlasAn/main/docs/build/html/index.html) and distributed under [Apache License 2.0](https://github.com/laitvo/PuPlasAn/blob/main/LICENSE). All details to particular functions and program modules are to be found in the above code scripts and in the example section, similarly to this module.

## Dependencies
The program scripts above are to be found in [*/rorap*](https://github.com/laitvo/RoRaP/tree/main/rorap) and are elaborated in [*/example*](https://github.com/laitvo/RoRaP/tree/main/example) folders, respectively. We recommend that the users download the example file alongside the program scripts.

Conventional libraries, `numpy`, `scipy`, and `pylab` are employed. Besides them, running the example script and conveniently using the library requires having [`spe2py`](https://pypi.org/project/spe2py/) and [`lmfit`](https://pypi.org/project/lmfit/) installed.

The code is distributed under Apache License 2.0. Authors welcome constructive feedback on its operability.
