# CompMatSciII-GPAW-Treatise

> **A Treatise on Computational Materials Science II**

> Dedicated to Prof. [George Kopidakis](https://www.materials.uoc.gr/faculty/kopidakis-giorgos/) and Prof. [Ioannis Remediakis](https://www.materials.uoc.gr/en/faculty/ioannis-remediakis/)
> — Exploring Density Functional Theory with GPAW and ASE —

---

## 📖 Table of Contents

1. [About This Repository](#about-this-repository)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [Assignments Overview](#assignments-overview)

   * [HW1: DFT Theory & Variational Framework](#hw1-dft-theory--variational-framework)
   * [HW2: Surface Energies & Wulff Constructions](#hw2-surface-energies--wulff-constructions)
   * [HW3: Mechanical Properties & Equations of State](#hw3-mechanical-properties--equations-of-state)
   * [HW4: Electronic Structure & Magnetic Ordering](#hw4-electronic-structure--magnetic-ordering)
   * [HW5: Vibrational Analysis & Phonons](#hw5-vibrational-analysis--phonons)
5. [TMDs Project](#tmds-project)
6. [Workflow & Dependencies](#workflow--dependencies)
7. [Contributing](#contributing)
8. [Acknowledgements](#acknowledgements)
9. [License](#license)

---

## About This Repository

This repository is a comprehensive compendium of my work in [**Computational Materials Science II**](https://mscs.uoc.gr/dmst/?courses=computational-materials-science-ii) as a first year graduate student. It embodies a deep dive into Density Functional Theory (DFT) using the **GPAW** code (real‐space projector augmented wave method) together with the **Atomic Simulation Environment (ASE)**. The focus spans—from the foundational Kohn–Sham and Hohenberg–Kohn theorems all the way to advanced applications such as band structures, Wulff constructions, density of states (DOS), mechanical equation of states, and phonon dispersions.

---

## High-level Repository Structure

```
.
├── CompMatSci_HW1/
│   ├── code/           # FPI solvers, numerical integrators, BFGS optimization scripts
│   ├── CompMatSci_HW_1.pdf      # Theory background & procedural framework
├── CompMatSci_HW2/
│   ├── code/       # Surface slab builders, dangling bond model, Wulff scripts
│   ├── results/        # Surface energy tables, Wulff shapes
│   ├── CompMatSci_HW_2.pdf     # Surface theory & equilibrium shape analysis
├── CompMatSci_HW3/
│   ├── code/           # Bulk‐phase optimizers, EOS fitting routines
│   ├── results/        # Energy vs. volume curves, EOS fits
│   ├── CompMatSci_HW_3.pdf      # Aluminum allotropes & WTe₂ isotherm study
├── CompMatSci_HW4/
│   ├── code/           # Band structure calculators, DOS & SOC/non-SOC scripts
│   ├── results/        # Band plots, DOS & magnetic ordering data
│   ├── CompMatSci_HW_4.pdf      # Electronic properties under DFT
├── CompMatSci_HW5/
│   ├── code/           # Diatomic vibration scripts, phonon dispersion pipelines
│   ├── results/        # Vibrational frequencies, dispersion curves
│   ├── CompMatSci_HW_5.pdf     # Vibrational analysis & phonon theory
├── README.md           # ← You are here  
└── LICENSE
```

Each `CompMatSci_HW*` directory contains:

* **`code/`**: Python scripts and GPAW/ASE calculators
* **`results/`** (where applicable): Raw output, plots, data tables
* **`CompMatSci_HW_?.pdf`**: A self-contained write-up detailing objectives, theoretical background, workflow, thought process, and discussion.

---

## Getting Started

### Prerequisites

* **Python 3.8+**
* **GPAW** (latest stable release)
* **ASE** (Atomic Simulation Environment)
* Typical Python packages: `numpy`, `scipy`, `matplotlib`, `pandas`

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YourUsername/CompMatSciII-GPAW-Treatise.git
   cd CompMatSciII-GPAW-Treatise
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install requirements:

   ```bash
   pip install gpaw ase numpy scipy matplotlib pandas
   ```

---

## Assignments Overview

### HW1: DFT Theory & Variational Framework

* **Topics**: Hohenberg–Kohn theorems, Kohn–Sham equations, variational principle
* **Procedures**: Fixed-point iteration (FPI), numerical integration techniques, BFGS optimization
* **Basis & Symmetry**: Plane-wave cut-off convergence, Monkhorst–Pack k-point sampling
* **Deliverables**: Convergence tests, code implementations, report with error analysis

### HW2: Surface Energies & Wulff Constructions

* **Topics**: Dangling bond model, slab constructions, surface energy estimation
* **Applications**: Wulff theorem for equilibrium crystal shapes
* **Deliverables**: Surface energy tables, 2D/3D Wulff shapes, discussion of facet stability

### HW3: Mechanical Properties & Equations of State

* **Systems**: Al (FCC & BCC), 2H-WTe₂
* **Methods**: Elbow method for PW and k-point hyperparameter selection
* **EOS Fits**: Murnaghan, Birch–Murnaghan, Vinet isotherms
* **Comparison**: Bulk modulus vs. literature values
* **Deliverables**: Energy–volume curves, EOS parameter tables, critical discussion

### HW4: Electronic Structure & Magnetic Ordering

* **Calculations**: Electronic band structures along high-symmetry paths
* **DOS**: Total and projected DOS
* **Magnetism**: SOC vs. non-SOC ground states, magnetic moment analysis
* **Deliverables**: Band/DOS plots, magnetic ordering summary, code notebooks

### HW5: Vibrational Analysis & Phonons

* **Scope**: Diatomic molecule vibrations; bulk phonon dispersion
* **Tools**: Finite difference force-constant method, GPAW phonon driver
* **Outputs**: Vibrational frequencies, dispersion curves along symmetry lines
* **Deliverables**: Frequency tables, dispersion plots, workflow documentation

---

## TMDs Project

The course project on transition-metal dichalcogenides (TMDs) electronic structure and DFT is maintained in the separate repository:

> **TMDs-Electronic-Structure-DFT**
> *[https://github.com/YourUsername/TMDs-Electronic-Structure-DFT](https://github.com/YourUsername/TMDs-Electronic-Structure-DFT)*

---

## Workflow & Dependencies

* **GPAW**: Real-space PAW DFT solver
* **ASE**: Structure builders, calculators, job management
* Scripts are modular: you can adapt them to new materials by editing the `input.py` files and convergence parameters.
* Reports explain directory structure, dependencies, and step-by-step execution.
Check the **Environment Setup** section of each assignment for case-specific dependencies and implementation nuances.
---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please adhere to the existing code style and include clear documentation for any new scripts.

---

## Acknowledgements

* Prof. George Kopidakis & Prof. Ioannis Remediakis for pioneering this engaging course.
* The GPAW development team for their powerful DFT package.
* The ASE community for continual support and expansion of simulation tools.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
Feel free to reuse and adapt the code for your own research and learning.
