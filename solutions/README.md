# Solutions

Before checking the solutions in this directory, please read the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf) for the details of the proposed method and the execution method.

## Settings

As we wrote in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf), we use different encoding and decoding methods, and different quantum error mitigation levels.

Here, the directory named `e0d0_*` uses the option of encoder optimization level 0 and decoder optimization level 0, which means the general encoding and decoding method described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

Similarly, the directory named `e2d1_*` uses the option of encoder optimization level 2 and decoder optimization level 1, which means the shallow encoding method for $|110\rangle$ and specific decoding method described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

Finally, the directory named `e2d2_*` uses the option of encoder optimization level 2 and decoder optimization level 2, which means the shallow encoding and decoding method for $|110\rangle$ described in the [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

## How to run the programs

Note that we are using [Mitiq](https://github.com/unitaryfund/mitiq) package for zero-noise extrapolation (ZNE).
Please install Mitiq by `pip install mitiq` before running the codes.

For each jupyter notebook, you can directly run all the cells from the first.

You can also check the previous results by changing the `filename` variable to the name of existing `job_ids_ibmq_jakarta_*.pkl` file.
Note that in this case, you do not have to run the cells for `execution` function and you can also skip the process to generate `*.pkl` files.

**Please first use [e2d2_qrem_zne/100step_jakarta.ipynb](https://github.com/BOBO1997/osp_solutions/blob/main/solutions/e2d2_qrem_zne/100step_jakarta.ipynb) to re-execute the solution.
This will output the result with high fidelity over 0.98.**

To see the behavior without QREM, please run the file [e2d2_qrem/100step_jakarta.ipynb](https://github.com/BOBO1997/osp_solutions/blob/main/solutions/e2d2_qrem/100step_jakarta.ipynb).
This will also output the result with high fidelity over 0.98.

To use the initial state other than $|110\rangle$, then please refer to the directories whose names start from `e0d0`.


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
