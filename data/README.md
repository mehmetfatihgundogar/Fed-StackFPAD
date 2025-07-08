### Data Preparation
In this paper, we conduct experiments on below datasets:
1. Replay-Attack (I)
2. Msu-Mfsd (M)
3. Oulu-Npu (O)
4. Casia-Fasd (C)

For instance; when you prepare data to configure the Replay-Attack, Oulu-Npu, and Casia-Fasd datasets as federated data centers and designate Msu-Mfsd as the test set, the script below will prepare the data accordingly.
```bash
./prepare_datasets.sh --base_folder /path/to/datasets --num_of_frame_steps 5 --training_datasets 1,3,4 --test_dataset 2
```

At that point the below folding structure will be created within the script:

```
fed-stackfpad__exp_mode_multiclient__train_on_IOC__test_on_M
|-- central
|-- 3_clients/
    |-- federated/
        |-- client_1.csv
        |-- client_2.csv
        |-- client_3.csv
|-- train
|-- validation
|-- test
|-- validation.csv
|-- test.csv
|-- labels.csv
```

The client_1.csv, client_2.csv, and client_3.csv files contain the filenames of the images belonging to each data center in the federated setup. 

**If you want to train the model using different datasets, you may modify the data preprocessing files in [Fed-StackFPAD/code/dataset_preprocessing](https://github.com/mgundogar/Fed-StackFPAD/tree/main/code/dataset_preprocessing). 


### Base ViT models
<table><tbody>
<th align="left">Base Model Name</th>
<th></th>
<!-- TABLE BODY -->
<tr>
<td>MAE weights pretrained on ImageNet-22k</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Base ViT-B/16 ViT Model</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>ViT-B/16 Model with ImageNet-22K Pretrained MAE Weights</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
</tbody></table>

&nbsp;
### Fine-tuned model checkpoints on target datasets

#### Single Data Center Models
<table><tbody>
<th align="left">Train Dataset</th>
<th align="left">Test Dataset</th>
<th align="left">Fine-tuned Model</th>
<!-- TABLE BODY -->
<tr>
<td>Oulu-Npu (O)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu (O)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu (O)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
</tbody></table>

&nbsp;
#### Centralized Models
<table><tbody>
<th align="left">Centralized Training Datasets (Aggregated in a Single Location)</th>
<th align="left">Test Dataset</th>
<th align="left">Fine-tuned Model</th>
<!-- TABLE BODY -->
<tr>
<td>Oulu-Npu + Casia-Fasd + Replay-Attack (O+C+I)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu + Msu-Mfsd + Replay-Attack (O+M+I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu + Casia-Fasd + Msu-Mfsd (O+C+M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Replay-Attack + Casia-Fasd + Msu-Mfsd (I+C+M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
</tbody></table>

&nbsp;
#### Federated Models
<table><tbody>
<th align="left">Train Datasets (Federated Data Centers)</th>
<th align="left">Test Dataset</th>
<th align="left">Fine-tuned Model</th>
<!-- TABLE BODY -->
<tr>
<td>Oulu-Npu, Casia-Fasd, Replay-Attack (O&C&I)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu, Msu-Mfsd, Replay-Attack (O&M&I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Oulu-Npu, Casia-Fasd, Msu-Mfsd (O&C&M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Replay-Attack, Casia-Fasd, Msu-Mfsd (I&C&M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
</tbody></table>

&nbsp;
#### Model Prediction Data for Stacked Ensemble
<table><tbody>
<th align="left">Prediction Fold</th>
<th align="left">Data Center Specific Model Prediction-1</th>
<th align="left">Data Center Specific Model Prediction-2</th>
<th align="left">Data Center Specific Model Prediction-1</th>
<th align="left">Federated Global Model Prediction</th>
<th align="left">Concatenated Prediction</th>
<!-- TABLE BODY -->
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com">O->M</a></td>
<td><a href="https://drive.google.com">C->M</a></td>
<td><a href="https://drive.google.com">I->M</a></td>
<td><a href="https://drive.google.com">O&C&I->M</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com">O->C</a></td>
<td><a href="https://drive.google.com">M->C</a></td>
<td><a href="https://drive.google.com">I->C</a></td>
<td><a href="https://drive.google.com">O&M&I->C</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com">O->I</a></td>
<td><a href="https://drive.google.com">C->I</a></td>
<td><a href="https://drive.google.com">M->I</a></td>
<td><a href="https://drive.google.com">O&C&M->I</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com">I->O</a></td>
<td><a href="https://drive.google.com">C->O</a></td>
<td><a href="https://drive.google.com">M->O</a></td>
<td><a href="https://drive.google.com">I&C&M->O</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com">O->M</a></td>
<td><a href="https://drive.google.com">C->M</a></td>
<td><a href="https://drive.google.com">I->M</a></td>
<td><a href="https://drive.google.com">O&C&I->M</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com">O->C</a></td>
<td><a href="https://drive.google.com">M->C</a></td>
<td><a href="https://drive.google.com">I->C</a></td>
<td><a href="https://drive.google.com">O&M&I->C</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com">O->I</a></td>
<td><a href="https://drive.google.com">C->I</a></td>
<td><a href="https://drive.google.com">M->I</a></td>
<td><a href="https://drive.google.com">O&C&M->I</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com">I->O</a></td>
<td><a href="https://drive.google.com">C->O</a></td>
<td><a href="https://drive.google.com">M->O</a></td>
<td><a href="https://drive.google.com">I&C&M->O</a></td>
<td><a href="https://drive.google.com">download</a></td>
</tr>
</tbody></table>


&nbsp;
#### SVM models created with Stacked Ensemble Training
<table><tbody>
<th align="left">Stacked Ensemble Model</th>
<!-- TABLE BODY -->
<tr>
<td><a href="https://drive.google.com">SVM Model - Target M</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com">SVM Model - Target C</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com">SVM Model - Target I</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com">SVM Model - Target O</a></td>
</tr>
</tbody></table>