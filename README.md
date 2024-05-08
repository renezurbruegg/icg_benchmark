<a name="readme-top"></a>


<!-- [![Paper][contributors-shield]][contributors-url]
[![ICG-Net Model][forks-shield]][forks-url]
[![Project Page][stars-shield]][stars-url] -->



<div align="center">

  <h3 align="center">Grasping Benchmarks</h3>

  <p align="center">
    VGN Benchmark environment for grasping in cluttered environments.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
[![MIT License][license-shield]][license-url]
[![Product Name Screen Shot][product-screenshot]](#)

This repo contains a easy to use and modular implementation of the benchmark environment for grasping in cluttered environments introduces by Breyer et. al [[1](https://github.com/ethz-asl/vgn)].
Note that we incoorperate slight modifications as done by Huang et. al [[2](https://github.com/HaojHuang/Edge-Grasp-Network)] to allow the network to skip and acquire a new viewpoint.


## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
#### Installation

**Installing the benchmark environment**
```bash
# Clone the repo

# Install the requirements
pip install -r requirements.txt

# Build the extension
python setup.py build_ext --inplace

# Install the package
pip install -e .

# Download the datasets and checkpoints
python scripts/download_data.py

```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Usage
#### Benchmark different Networks

**ICG-Net**
```bash
# Evaluate in packed scene
python scripts/test_icgnet.py --scene packed --object-set packed/test

# Evaluate in pile scene
python scripts/test_icgnet.py --scene pile --object-set pile/test
```

**Edge-Grasp Network**
```bash 
# Evaluate in packed scene
python scripts/test_edge.py --method edge-vn --scene packed --object-set packed/test

# Evaluate in pile scene
python scripts/test_edge.py --method edge-vn --scene pile --object-set pile/test
```
**VN-Edge-Grasp Network**
```bash
# Evaluate in packed scene
python scripts/test_edge.py --method edge-vn --scene packed --object-set packed/test

# Evaluate in pile scene
python scripts/test_edge.py --method edge-vn --scene pile --object-set pile/test
```

**GIGA Network**
```bash
# Evaluate in packed scene
python scripts/test_giga.py --scene packed --object-set packed/test

# Evaluate in pile scene
python scripts/test_giga.py --scene pile --object-set pile/test
```



## Usage
 
 Will be updated soon.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the BSD-2 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Citing
If you use this code in your research, please cite the following paper:
```
@article{zurbrugg2024icgnet,
  title={ICGNet: A Unified Approach for Instance-Centric Grasping},
  author={Zurbr{\"u}gg, Ren{\'e} and Liu, Yifan and Engelmann, Francis and Kumar, Suryansh and Hutter, Marco and Patil, Vaishakh and Yu, Fisher},
  journal={arXiv preprint arXiv:2401.09939},
  year={2024}
}
```

Also consider citing the original work by Breyer et. al that introduced the benchmark environment:
```
@inproceedings{breyer2020volumetric,
 title={Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter},
 author={Breyer, Michel and Chung, Jen Jen and Ott, Lionel and Roland, Siegwart and Juan, Nieto},
 booktitle={Conference on Robot Learning},
 year={2020},
}
```


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew -->

[license-url]: https://github.com/renezurbruegg/ICG-Net/blob/master/LICENSE.txt
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[product-screenshot]: docs/imgs/env_example.png

<!-- [Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com  -->
