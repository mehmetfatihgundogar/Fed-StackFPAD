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
<td><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
</tr>
<tr>
<td>Base ViT-B/16 ViT Model</td>
<td><a href="https://drive.google.com/file/d/1ikAlNhyUVP3qSawhVOUnyb9N72rYzFpd/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>ViT-B/16 Model with ImageNet-22K Pretrained MAE Weights</td>
<td><a href="https://drive.google.com/file/d/1UxvIP6F9pcJQw0cxG4zF2VrS5JCD_tOr/view?usp=sharing">download</a></td>
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
<td><a href="https://drive.google.com/file/d/1A0x8ZviJ6-TrquTU-2ZthEiWFrpXPp88/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com/file/d/1mrIx-0KX0mDAcuGh6IDehmMXHYzI5NfD/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Msu-Mfsd (M)</td>
<td><a href="https://drive.google.com/file/d/1IJS4AL6F4ZNpYdAbufj8fLB5XiUmS2UE/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu (O)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com/file/d/1bDgSUE-eYRr6_Sid-y2_MFYB1YhIKyYc/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com/file/d/1Vxlrv_6uYWXdHrAgoHbb77twYbEBIujp/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com/file/d/1ruodMsNEC3GDE1Wd-ZSzR_dDVzQ3gvkG/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu (O)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com/file/d/1JmNbpd18fYUC4h1gassyxY6Lf5b_NQhg/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com/file/d/1MXrA0RDiltyPSy_nITI0HLFipVMKVqaz/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com/file/d/1a7DBSuEwGKib6bPfIE6I-iJMjhFQU21H/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Msu-Mfsd (M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com/file/d/1GrNEXZtF7433url_RTkHnCdsC9bifYEP/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Casia-Fasd (C)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com/file/d/1z8AZCzRyhCceFpVLfXwylTX01Pi-_77o/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Replay-Attack (I)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com/file/d/1G0Of4ucWclWI6bvbFtDr2Kk6eQdmieCv/view?usp=sharing">download</a></td>
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
<td><a href="https://drive.google.com/file/d/1jkZX_334wXssC5zqldS7IM6IPrTxd-qF/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu + Msu-Mfsd + Replay-Attack (O+M+I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com/file/d/1Xng9G0PBANA_tUQh9W6NSm55Q3inhIiJ/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu + Casia-Fasd + Msu-Mfsd (O+C+M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com/file/d/1KgKgZR-Duzv9awDhm5eOSfBRHYn7PyhF/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Replay-Attack + Casia-Fasd + Msu-Mfsd (I+C+M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com/file/d/1XEex2T9jdAdf-u_-NweQIhWp5_0zrSjA/view?usp=sharing">download</a></td>
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
<td><a href="https://drive.google.com/file/d/1J7L1LrNgIKFUfPEoHCVcmKzduGGjKEGl/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu, Msu-Mfsd, Replay-Attack (O&M&I)</td>
<td>Casia-Fasd (C)</td>
<td><a href="https://drive.google.com/file/d/1bUTtSfCcG7vdREKUdfqA52Lv10224Urr/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Oulu-Npu, Casia-Fasd, Msu-Mfsd (O&C&M)</td>
<td>Replay-Attack (I)</td>
<td><a href="https://drive.google.com/file/d/1vcvzVllUzrMj_sjSdRVf1DlgQYVutcMQ/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Replay-Attack, Casia-Fasd, Msu-Mfsd (I&C&M)</td>
<td>Oulu-Npu (O)</td>
<td><a href="https://drive.google.com/file/d/1W7dKKD11mOyug-I_JZBYAIywat-C9E-J/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

&nbsp;
#### Model Prediction Data for Stacked Ensemble
<table><tbody>
<th align="left">Prediction Fold</th>
<th align="left">Data Center Specific Model Prediction-1</th>
<th align="left">Data Center Specific Model Prediction-2</th>
<th align="left">Data Center Specific Model Prediction-3</th>
<th align="left">Federated Global Model Prediction</th>
<th align="left">Concatenated Prediction</th>
<!-- TABLE BODY -->
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com/file/d/1v78w8wprqXTeIdsLUL2Fyhxz29lAR-oO/view?usp=sharing">O->M</a></td>
<td><a href="https://drive.google.com/file/d/1rjq_L_uu_MQym01esJRL5wNMyorWl9cK/view?usp=sharing">C->M</a></td>
<td><a href="https://drive.google.com/file/d/1VKVPFhBk3iRU-0RjmPteLuWnKzxi6230/view?usp=sharing">I->M</a></td>
<td><a href="https://drive.google.com/file/d/1xnxTMOJd4NRxKMO-wHLnPSwFSSf5bLeD/view?usp=sharing">O&C&I->M</a></td>
<td><a href="https://drive.google.com/file/d/151NVjDbsrIGAYivZQJmqqjDEqesap50K/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com/file/d/1JG3M2-5B9HX4IPRbtbrZyYHHpTuYCqLx/view?usp=sharing">O->C</a></td>
<td><a href="https://drive.google.com/file/d/1MzbpIcaDb6PoM8i0zeBuhjRsUX-xGMVq/view?usp=sharing">M->C</a></td>
<td><a href="https://drive.google.com/file/d/16t6RGHonWT5gnxTf_MCctp6x1Xh0CONr/view?usp=sharing">I->C</a></td>
<td><a href="https://drive.google.com/file/d/1IIqbCdW-4Qfkd2jpHBoQFuu3KbkCT2XZ/view?usp=sharing">O&M&I->C</a></td>
<td><a href="https://drive.google.com/file/d/160qz1bLe4s7QlldQ_brsHiIRb-pGsGft/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com/file/d/1pqsbka7Vx9rFbE5GRTJNNJkRScY8xUMu/view?usp=sharing">O->I</a></td>
<td><a href="https://drive.google.com/file/d/1FLLTrHaVcwvclBtVicqKexvDl5FguV48/view?usp=sharing">C->I</a></td>
<td><a href="https://drive.google.com/file/d/16Mpvcwo-pwnQURHcaXRPLjWPw4-_Bku5/view?usp=sharing">M->I</a></td>
<td><a href="https://drive.google.com/file/d/1PDtRiuvrpvOzE5DVBfnadCIgXdFaBN2I/view?usp=sharing">O&C&M->I</a></td>
<td><a href="https://drive.google.com/file/d/1glsacsnM1rXEDJvTY2QmUM-dH8jZ9XAk/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Validation</td>
<td><a href="https://drive.google.com/file/d/1gNuVyF3TWW31pNOal6NlxlCQsvvrPNcE/view?usp=sharing">I->O</a></td>
<td><a href="https://drive.google.com/file/d/1M-gRABfHB9Fe1hwzHMs8g94sXvGBvpc5/view?usp=sharing">C->O</a></td>
<td><a href="https://drive.google.com/file/d/1oG9ToX-ayU3a_0cLdYKETuEHB_sC7J7i/view?usp=sharing">M->O</a></td>
<td><a href="https://drive.google.com/file/d/1PXpniNp7Uhtwoh008wUfjv2hDmx-eKCD/view?usp=sharing">I&C&M->O</a></td>
<td><a href="https://drive.google.com/file/d/1wCGpCHRr3yxQF3SZfdHh0P0Vc4hjqYO4/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com/file/d/1hUGd8tTdd9EhYBu4hMAc5jDnpz37XYfZ/view?usp=sharing">O->M</a></td>
<td><a href="https://drive.google.com/file/d/1LfOIIyFVeROBlWP-rpgd8ZhqPo-VWAZG/view?usp=sharing">C->M</a></td>
<td><a href="https://drive.google.com/file/d/13u5LO0FKrD4rN747tpjGj30PYVNPQsKn/view?usp=sharing">I->M</a></td>
<td><a href="https://drive.google.com/file/d/187AP2NZLWrlxnenwZd09_DdCku71a6il/view?usp=sharing">O&C&I->M</a></td>
<td><a href="https://drive.google.com/file/d/16mXspp6zv-EUpXDeoOF_3t67oPr_kR7-/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com/file/d/1zK8Qv59LMqVF_EOuIiIAnYZpTO5rkWE7/view?usp=sharing">O->C</a></td>
<td><a href="https://drive.google.com/file/d/18qyHsXE1V6hwzviE0HeskQVjhogvNsXw/view?usp=sharing">M->C</a></td>
<td><a href="https://drive.google.com/file/d/1MpMBlDK0IkMTReYzBzHjybcYcUzmBOLh/view?usp=sharing">I->C</a></td>
<td><a href="https://drive.google.com/file/d/1K6zHe5qA1V-1KydKgVV0KlS2613TblgN/view?usp=sharing">O&M&I->C</a></td>
<td><a href="https://drive.google.com/file/d/1Ux0XxW8Ju4YUSOEXUrxYo82w32QWRGR-/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com/file/d/1RbYq5FY0f33bMjpvyQt18qk_GDaDe_lH/view?usp=sharing">O->I</a></td>
<td><a href="https://drive.google.com/file/d/15fwApZSdDuosv8p3_dGOSi_Sw1_WNYPh/view?usp=sharing">C->I</a></td>
<td><a href="https://drive.google.com/file/d/1SgBkzLKlEt6RLoZcVlDTs3T8g65OKJGy/view?usp=sharing">M->I</a></td>
<td><a href="https://drive.google.com/file/d/1R8wpv63j0qF6297aEUYdiAIHjgTxvTPb/view?usp=sharing">O&C&M->I</a></td>
<td><a href="https://drive.google.com/file/d/1w5VJK0-sh0gEWUAgAlEwqpAp_nycl7MI/view?usp=sharing">download</a></td>
</tr>
<tr>
<td>Test</td>
<td><a href="https://drive.google.com/file/d/1e8CXqBgXdCbv622Q0kgRiKfNGu0NW-df/view?usp=sharing">I->O</a></td>
<td><a href="https://drive.google.com/file/d/159DR_54JsOBzZggk0cZddDXiMeOwYA-l/view?usp=sharing">C->O</a></td>
<td><a href="https://drive.google.com/file/d/1vo0mKXNKtbpLuEc-x3Xm9yRjrPHVOtxA/view?usp=sharing">M->O</a></td>
<td><a href="https://drive.google.com/file/d/1CoGhpiVbPzJeWeUyvuUSKrqhigXpieaf/view?usp=sharing">I&C&M->O</a></td>
<td><a href="https://drive.google.com/file/d/1HpjwxVCzo2AxkocJm-bctH9AYs0uN7TL/view?usp=sharing">download</a></td>
</tr>
</tbody></table>


&nbsp;
#### SVM models created with Stacked Ensemble Training
<table><tbody>
<th align="left">Stacked Ensemble Model</th>
<!-- TABLE BODY -->
<tr>
<td><a href="https://drive.google.com/file/d/1xf8Pglsn3NDT2OXOmQ2GJalq6ibAQ0Ym/view?usp=sharing">SVM Model - Target M</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com/file/d/15_6081XlZ-0Jg_ansDZn1w4msKKW0-hM/view?usp=sharing">SVM Model - Target C</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com/file/d/1AeN-YddZW8oNLoT_JxdVvZmLymHUveuR/view?usp=sharing">SVM Model - Target I</a></td>
</tr>
<tr>
<td><a href="https://drive.google.com/file/d/1tQGBleTIt5peD3yuUEUP3YQh8b-Zii3v/view?usp=sharing">SVM Model - Target O</a></td>
</tr>
</tbody></table>