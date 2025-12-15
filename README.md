# Robotic Action Frame Prediction
- Report Link: [Here](report.pdf)
- Data generation:  
    - Robotwin: [this link](https://igbmq7q8vt.feishu.cn/docx/VJCzdPUEyonbFIxo1HhceXGAnhg?from=from_copylink)  
    - pkl to hdf5:  
    ```
    python pkl2hdf5_rdt.py ${task_name} ${setting} ${expert_data_num}
    ```    
    reference: [this link](https://zhuanlan.zhihu.com/p/23846224178?utm_source=wechat_session&utm_medium=social&s_r=0)
- Environment setup:   
    ```
    pip install -r requirements.txt
    ```
- run training:   
    ```
    bash train.sh
    ``` 
    or  
    ```
    bash train2.sh
    ```  
    the training parametets of 2 scripts are different.
- evaluation:  
    ```
    python -u evaluation.py
    python -u summary.py
    ```
    evaluation results are in [results ](results)folder.
---
