# Solutions

Before checking the solutions in this directory, please read the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf) for the details of the proposed method and the execution method.

## Settings

As we wrote in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf), we use different encoding and decoding methods, and different quantum error mitigation levels.

Here, the directory named `e0d0_*` uses the option of encoder optimization level 0 and decoder optimization level 0, which means the general encoding and decoding method described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

Similarly, the directory named `e2d1_*` uses the option of encoder optimization level 2 and decoder optimization level 1, which means the shallow encoding method for $|110\rangle$ and specific decoding method described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

Finally, the directory named `e2d2_*` uses the option of encoder optimization level 2 and decoder optimization level 2, which means the shallow encoding and decoding method for $|110\rangle$ described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

The level of the error mitigation is also identified by the name of directory.
- `*_qrem` directory: The results with quantum readout error mitigation (QREM)
- `*_qrem_zne` directory: The results with QREM and zero-noise extrapolation (ZNE)
- `*_qrem_zne_pt` directory: The results with QREM and ZNE and Pauli twirling

That is, if you are looking for the results by "Shallow encoding and specific decoding with QREM and ZNE", you can see the files in `e2d1_qrem_zne` directory.

Each directory contains
- `100steps_fake.ipynb`, which runs the designated implementation setting on `fake_jakarta` simulator.
- `100steps_jakarta.ipynb`, which runs the designated implementation setting on `ibmq_jakarta` real device.

In the `*_qrem` directory, you can find the `100steps_*_raw.ipynb` files. They are just removing the QREM operation in the fidelity calculation to see the raw results without error mitigation.

## Results and Corresponding Directories

We made the TABLE I in [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf) by the source code in the following directories.

Setting | Score (`fake_jakarta`) | Score (`ibmq_jakarta`) | Directory |
--- | --- | --- | ---
**General encoding and general decoding** |  |  |
without any QEM | 0.7856 ± 0.0015 | 0.8039 ± 0.0048 | `e0d0_qrem`
with QREM | 0.8448 ± 0.0015 | 0.9032 ± 0.0054 | `e0d0_qrem`
with QREM, ZNE | 0.9393 ± 0.0053 | 0.9866 ± 0.0017 | `e0d0_qrem_zne_2`
with QREM, ZNE and Pauli Twirling | 0.9801 ± 0.0031 |  | `e0d0_qrem_zne_pt`
**Shallow encoding and specific decoding** |  |  |
without any QEM | 0.8631 ± 0.0017 | 0.8637 ± 0.0041 | `e2d1_qrem`
with QREM | 0.9234 ± 0.0016 | 0.9728 ± 0.0040 | `e2d1_qrem`
with QREM, ZNE | 0.9840 ± 0.0024 | 0.9857 ± 0.0043 | `e2d1_qrem_zne`
with QREM, ZNE and Pauli Twirling | 0.9714 ± 0.0048 | 0.9624 ± 0.0167 | `e2d1_qrem_zne_pt`
**Shallow encoding and shallow decoding** |  |  |
without any QEM | 0.8863 ± 0.0012 | 0.8803 ± 0.0044 | `e2d2_qrem`
with QREM | 0.9533 ± 0.0017 | 0.9852 ± 0.0061 | `e2d2_qrem`
with QREM, ZNE | 0.9855 ± 0.0036 | 0.9929 ± 0.0015 | `e2d2_qrem_zne`
with QREM, ZNE and Pauli Twirling | 0.9801 ± 0.0031 | 0.9768 ± 0.0034 | `e2d2_qrem_zne_pt`

In the table above, the number of Trotter steps is fixed to 100.

In `e2d2_qrem_20220412`, we further scored the fidelity 0.9928 ± 0.0013 with 15 Trotter steps and only with QREM.

## How to Run (Re-execute) the Programs

Note that we are using [Mitiq](https://github.com/unitaryfund/mitiq) package for zero-noise extrapolation (ZNE).
Please install Mitiq by `pip install mitiq` before running the codes.

For each jupyter notebook, you can directly run all the cells from the first.

You can also check the previous results by changing the `filename` variable to the name of existing `job_ids_ibmq_jakarta_*.pkl` file.
Note that in this case, you do not have to run the cells for `execution` function and you can also skip the process to generate `*.pkl` files.

**Please first use [e2d2_qrem_zne/100step_jakarta.ipynb](https://github.com/BOBO1997/osp_solutions/blob/main/solutions/e2d2_qrem_zne/100step_jakarta.ipynb) to re-execute and evaluate the solution.
This will output the result with high fidelity over 0.98.**

To see the behavior without QREM, please run the file [e2d2_qrem/100step_jakarta.ipynb](https://github.com/BOBO1997/osp_solutions/blob/main/solutions/e2d2_qrem/100step_jakarta.ipynb).
This will also output the result with high fidelity over 0.98.

To use the initial state other than $|110\rangle$, then please refer to the directories whose names start from `e0d0`.
Note that the general encoding and deconding setting with QREM and ZNE will also output the fidelity over 0.98.

We strongly believe the optimization algorithm (level 3) provided by the `qiskit.compiler.transpile` function will always reduce the quantum circuit with 100 trotter steps to the constant depth cirucit, but if you found any trouble in reproducibility, please contact us.
At least we have already casted and retrieved the jobs with those successfully optimized circuits, which support such optimization algorithm does exists.

##  Directory Structure
```bash
.
├── README.md
├── e0d0_qrem
│   ├── 100step_fake.ipynb
│   ├── 100step_fake_raw.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── 100step_jakarta_raw.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_132447_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_132447_.pkl
│   └── properties_ibmq_jakarta_20220413_132447_.pkl
├── e0d0_qrem_zne
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_152244_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_152244_.pkl
│   └── properties_ibmq_jakarta_20220413_152244_.pkl
├── e0d0_qrem_zne_2
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_180241_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_180241_.pkl
│   └── properties_ibmq_jakarta_20220413_180241_.pkl
├── e0d0_qrem_zne_pt
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220417_134139_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220417_134139_.pkl
│   └── properties_ibmq_jakarta_20220417_134139_.pkl
├── e2d1_qrem
│   ├── 100step_fake.ipynb
│   ├── 100step_fake_raw.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── 100step_jakarta_raw.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030038_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030038_.pkl
│   └── properties_ibmq_jakarta_20220413_030038_.pkl
├── e2d1_qrem_20220412
│   ├── 15step_fake.ipynb
│   ├── 15step_jakarta.ipynb
│   ├── 15step_jakarta_raw.ipynb
│   ├── job_ids_jakarta_15step_20220412_031437_.pkl
│   ├── jobs_jakarta_15step_20220412_031437_.pkl
│   └── properties_jakarta20220412_031437_.pkl
├── e2d1_qrem_zne
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030253_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030253_.pkl
│   └── properties_ibmq_jakarta_20220413_030253_.pkl
├── e2d1_qrem_zne_pt
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030456_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030456_.pkl
│   └── properties_ibmq_jakarta_20220413_030456_.pkl
├── e2d2_qrem
│   ├── 100step_fake.ipynb
│   ├── 100step_fake_raw.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── 100step_jakarta_raw.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030136_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030136_.pkl
│   └── properties_ibmq_jakarta_20220413_030136_.pkl
├── e2d2_qrem_20220412
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── 100step_jakarta_raw.ipynb
│   ├── circuit_utils.py
│   ├── job_ids_jakarta_100step_20220412_171248_.pkl
│   ├── jobs_jakarta_100step_20220412_171248_.pkl
│   ├── properties_jakarta20220412_171248_.pkl
│   ├── sgs_algorithm.py
│   ├── tomography_utils.py
│   └── zne_utils.py
├── e2d2_qrem_zne
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030626_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030626_.pkl
│   └── properties_ibmq_jakarta_20220413_030626_.pkl
├── e2d2_qrem_zne_pt
│   ├── 100step_fake.ipynb
│   ├── 100step_jakarta.ipynb
│   ├── job_ids_ibmq_jakarta_100step_20220413_030821_.pkl
│   ├── jobs_ibmq_jakarta_100step_20220413_030821_.pkl
│   └── properties_ibmq_jakarta_20220413_030821_.pkl
└── utils
    ├── circuit_utils.py
    ├── sgs_algorithm.py
    ├── tomography_utils.py
    └── zne_utils.py
```
