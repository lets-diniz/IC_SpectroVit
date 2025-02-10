import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import random
random.seed(5)
from scipy import signal,stats
import os
import seaborn as sns
import json

from models import SpectroViT
from utils import read_yaml, clean_directory
from utils_for_evaluation import *

"""
Script to test and compare diferent versions of SpectroVit's model trained varing hop
value used in STFT.
Receives an input YAML file with test description.
When the script finishes runing, one will find a folder with txt file with quantitative results and 
tables comparing each model reconstruction capabilities. The folder will also contain pictures with examples
of reconstructed spectra, their histograms and heatmaps to visually compare their quantitative performance.
Models can be evaluated considering only the hop value each one was trained with, or we can also evaluate
them in all hop values defined in the YAML file, to capture the models capabilites of generalization.
"""

#---------------------Definitions---------------------------
config_path = input("Enter path for config.yaml: ")
config = read_yaml(file=config_path)
device = torch.device(config['device'])

#name for test folder
name_test = str(config['test_name'])
path_save_test = '../'+name_test+'/'
os.makedirs(path_save_test, exist_ok=True)
clean_directory(dir_path=path_save_test)

#hop size used for each model compared
hop_size = list(config['hop_values'])
window_size = 256
window = signal.windows.hann(256,sym = True)

#path to models being compared
list_model_names = list(config['model_names'])
list_models = []
for idx, model in enumerate(list_model_names):
    list_models.append(SpectroViT().to(device))
    list_models[idx].load_state_dict(torch.load(list_model_names[idx],map_location=config['device']))

#nicknames for each model for reference in tables and result figures
name_models = list(config['model_nicknames'])
#path to test data
path_to_test_data = str(config['test_dataset']['path'])
#dataset type for evaluation: can be different for each model or the same for all
if not isinstance(config['test_dataset']['type'], list):
    dataset_list = []
    for i in range(len(list_models)):
        dataset_list.append(config['test_dataset']['type'])
else:
    dataset_list=config['test_dataset']['type']

#show results evaluating each model on the hop it was trained for
show_results_for_each_hop = bool(config.get('show_results_for_each_hop',False))
#show results evaluating each model in all hop values in hop_size list
show_results_for_all_hops_combined = bool(config.get('show_results_for_all_hops_combined',False))

#-------------------------------Inferences---------------------------------------
print('inference for the metrics')
#get quantitative results
dict_metrics = get_metrics_for_different_models(path_to_test_data=path_to_test_data,
                                                list_models=list_models,
                                                name_models=name_models,
                                                hop_size=hop_size,
                                                window_size=window_size,
                                                window=window,
                                                device=device,
                                                dataset_list=dataset_list)

print('inference for examples')
#make predictions to plot figures with results and acquire stastical information about reconstruction
predictions, target_concat, ppm_concat,input_spgrams = get_inference_instances(path_to_test_data=path_to_test_data,
                                                                               list_models=list_models,
                                                                               name_models=name_models,
                                                                               hop_size=hop_size,
                                                                               window_size=window_size,
                                                                               window=window,
                                                                               device=device,
                                                                               dataset_list=dataset_list)

with open(path_save_test+"quantitative_metrics.txt", "w") as f:
    f.write("dict_metrics=")
    f.write(json.dumps(dict_metrics,indent=4))

#-----------------------------histograms---------------------------------
#histogram of spectrum reconstructions obtained by each model
for i,hop in enumerate(hop_size):
    if len(name_models) == len(hop_size):
        plt.hist(predictions[str(int(hop))][name_models[i]].flatten(),bins=100,density=True,alpha=0.35,label=name_models[i])
    else:
        for name in name_models:
            plt.hist(predictions[str(int(hop))][name].flatten(),bins=100,density=True,alpha=0.35,label=name+'and hop'+str(int(hop)))
plt.hist(target_concat.flatten(),bins=100,density=True,alpha=0.35,color='black',label='target')
plt.yscale('log')
plt.legend(loc='upper right')
plt.title('Histogram Predictions and Target')
plt.savefig(path_save_test+"histograms.png")
plt.close()

#build table with range of spectrum reconstructions 
#max and min values in reconstructed spectra for each model
str_aux=''
str_aux=str_aux+'range target: '+str(np.min(target_concat))+'/'+str(np.max(target_concat))+'\n'
for i,hop in enumerate(hop_size):
    if len(name_models) == len(hop_size):
        str_aux=str_aux+'range predictions '+name_models[i]+':'+str(np.min(predictions[str(int(hop))][name_models[i]]))+'/'+str(np.max(predictions[str(int(hop))][name_models[i]]))+'\n'
    else:
        for name in name_models:
            str_aux=str_aux+'range predictions '+name+' and hop'+str(int(hop))+':'+str(np.min(predictions[str(int(hop))][name_models[i]]))+'/'+str(np.max(predictions[str(int(hop))][name]))+'\n'
str_aux=str_aux+'\n'
with open(path_save_test+"extra_metrics.txt", "w") as f:
    f.write("range_predictions:\n")
    f.write(str_aux)

#plot histogram figure
for i,hop in enumerate(hop_size):
    if len(name_models) == len(hop_size):
        data_sorted_pred = np.sort(predictions[str(int(hop))][name_models[i]].flatten())
        cumulative_pred = np.arange(1, len(data_sorted_pred) + 1) / len(data_sorted_pred) 
        plt.plot(data_sorted_pred, cumulative_pred, linestyle='dotted',label=name_models[i])
    else:
        for name in name_models:
            data_sorted_pred = np.sort(predictions[str(int(hop))][name].flatten())
            cumulative_pred = np.arange(1, len(data_sorted_pred) + 1) / len(data_sorted_pred) 
            plt.plot(data_sorted_pred, cumulative_pred, linestyle='dotted',label=name+'and hop'+str(int(hop)))
  
data_sorted_tgt = np.sort(target_concat.flatten())
cumulative_tgt = np.arange(1, len(data_sorted_tgt) + 1) / len(data_sorted_tgt)
plt.plot(data_sorted_tgt, cumulative_tgt, linestyle='dotted',color='black',label='target')
plt.title('Cumulated Histogram of Predictions and Target')
plt.legend(loc='lower right')
plt.savefig(path_save_test+"histograms_cumulated.png")
plt.close()

#----------------------------stats--------------------------------------------------
#get mean/std/median/skewness/kurtosis of reconstructed spectra of each model
tgt_stats,pred_stats = get_stats_from_predictions(name_models=name_models,
                                                  hop_size=hop_size,
                                                  target_concat=target_concat,
                                                  predictions=predictions)
#write results in table
str_aux=''
str_aux = '|..............Mean.......|...........STD..........|..........Median........|..............Skew.......|..........Kurtosis......|'+'\n'
for i,key in enumerate(list(tgt_stats.keys())):
    if i == 0:
        str_aux=str_aux+'| '+ 'target        ' + '{:.4E}'.format(tgt_stats[key])+ '|'
    else:
        str_aux=str_aux+ 'target        ' + '{:.4E}'.format(tgt_stats[key])+ '|'
str_aux=str_aux+'\n'
for i,hop in enumerate(hop_size):
    if len(name_models) == len(hop_size):
        for j,key in enumerate(list(tgt_stats.keys())):
            if j == 0:
                str_aux=str_aux+'| '+name_models[i]+'        '+ '{:.4E}'.format(pred_stats[key][i])+ '|'
            else:
                str_aux=str_aux+name_models[i]+'        '+ '{:.4E}'.format(pred_stats[key][i])+ '|'
        str_aux=str_aux+'\n'
    else:
        for k,name in enumerate(name_models):
            for j,key in enumerate(list(tgt_stats.keys())):
                if j == 0:
                    str_aux=str_aux+'| '+name+' and hop '+str(int(hop))+' '+ '{:.4E}'.format(pred_stats[key][i*len(name_models)+k])+ '|'
                else:
                    str_aux=str_aux+name+' and hop '+str(int(hop))+' '+ '{:.4E}'.format(pred_stats[key][i*len(name_models)+k])+ '|'
            str_aux=str_aux+'\n'
    #print average and std of stats
    if i == len(hop_size)-1:
        for j,key in enumerate(list(tgt_stats.keys())):
            if j == 0:
                str_aux=str_aux+'| '+'avg. models'+'   '+ '{:.4E}'.format(np.mean(pred_stats[key]))+ '|'
            else:
                str_aux=str_aux+'avg. models'+'   '+ '{:.4E}'.format(np.mean(pred_stats[key]))+ '|'
        str_aux=str_aux+'\n'
        for j,key in enumerate(list(tgt_stats.keys())):
            if j == 0:
                str_aux=str_aux+'| '+'std. models'+'   '+ '{:.4E}'.format(np.std(pred_stats[key]))+ '|'
            else:
                str_aux=str_aux+'std. models'+'   '+ '{:.4E}'.format(np.std(pred_stats[key]))+ '|'
        str_aux=str_aux+'\n'
str_aux=str_aux+'\n'
with open(path_save_test+"extra_metrics.txt", "a") as f:
    f.write("stats_predictions:\n")
    f.write(str_aux)


#-----------------------------------metrics and score tables-------------------------------------------------

metrics_names = ['Loss','MSE','SNR','FWHM','ShSc']

#order models by quantitative performance
if len(name_models) == len(hop_size):
    #----------------------------metrics_per_model--------------------------------------------------
    results_for_their_right_hop = get_performance_metric_for_trained_hop(name_models=name_models,
                                                                               hop_size=hop_size,
                                                                               dict_metrics=dict_metrics,
                                                                               metrics_names=metrics_names)
    #----------------------------score_per_model--------------------------------------------------
    models_scored_for_their_right_hop, models_score_value_for_their_right_hop = get_scores_for_given_hop(name_models=name_models,
                                                                                                            metrics_names=['MSE','SNR','FWHM','ShSc'],
                                                                                                            results_for_given_hop=results_for_their_right_hop)
    #write table with metrics and score (from best to worse model in each category)
    str_aux=''
    str_aux=str_aux+'|.........Loss........|..........MSE........|..........SNR........|.........FWHM........|.........ShSc........|Score wto Loss|'+'\n'
    for line_idx in range(len(name_models)):
        for metric_idx,metric in enumerate(metrics_names):
            if metric_idx == 0:
                str_aux = str_aux+ '| '+results_for_their_right_hop[metric][0][line_idx]+' | '+'{:.4E}'.format(results_for_their_right_hop[metric][1][line_idx])+' | '
            else:
                str_aux = str_aux + results_for_their_right_hop[metric][0][line_idx]+' | '+'{:.4E}'.format(results_for_their_right_hop[metric][1][line_idx])+' | '
        if len(str(models_score_value_for_their_right_hop[line_idx])) == 2:
            space='   | '
        else:
            space='    | '
        str_aux=str_aux+models_scored_for_their_right_hop[line_idx] + ': '+str(models_score_value_for_their_right_hop[line_idx])+space+'\n'    
    #print average and std of models for each metric
    for metric_idx,metric in enumerate(metrics_names):
        if metric_idx == 0:
            str_aux=str_aux+'| '+'avg.'+'   | '+ '{:.4E}'.format(np.mean(results_for_their_right_hop[metric][1]))+ ' |'
        else:
            str_aux=str_aux+' avg.'+'   | '+ '{:.4E}'.format(np.mean(results_for_their_right_hop[metric][1]))+ ' |'
    str_aux=str_aux+'\n'
    for metric_idx,metric in enumerate(metrics_names):
        if metric_idx == 0:
            str_aux=str_aux+'| '+'std.'+'   | '+ '{:.4E}'.format(np.std(results_for_their_right_hop[metric][1]))+ ' |'
        else:
            str_aux=str_aux+' std.'+'   | '+ '{:.4E}'.format(np.std(results_for_their_right_hop[metric][1]))+ ' |'
    str_aux=str_aux+'\n'+'\n'
    with open(path_save_test+"extra_metrics.txt", "a") as f:
        f.write("metrics_and_score_for_trained_hop:\n")
        f.write(str_aux)


    #----------------------------noise and GABA peak per hop--------------------------------------------------
    #write table with mean noise estimation and GABA peak height in reconstructed spectra of each model
    str_aux=''
    for i,hop in enumerate(hop_size):
        str_aux=str_aux+'noise estimation on prediction of '+name_models[i]+':' + '{:.4E}'.format(noise_est(ppm_concat,predictions[str(int(hop))][name_models[i]]))+' / '+'GABA peak:'+'{:.4E}'.format(get_max_gaba(ppm_concat,predictions[str(int(hop))][name_models[i]]))+'\n'
    str_aux=str_aux+'\n'
    with open(path_save_test+"extra_metrics.txt", "a") as f:
            f.write("noise_and_GABA_peak:\n")
            f.write(str_aux)


#get score at each metric for each model for each value of hop in hop_size list
results_for_each_hop = get_results_for_each_hop(name_models=name_models,
                                                hop_size=hop_size,
                                                metrics_names=metrics_names,
                                                dict_metrics=dict_metrics)
if show_results_for_each_hop is True:
    #-----------------------------------metrics per hop------------------------------------------------
    for hop in hop_size:
        models_scored_for_given_hop, models_score_value_for_given_hop = get_scores_for_given_hop(name_models=name_models,
                                                                                                 metrics_names=['MSE','SNR','FWHM','ShSc'],
                                                                                                 results_for_given_hop=results_for_each_hop[str(int(hop))])
        #write result metrics for each model for current hop value
        str_aux=''
        str_aux=str_aux+'|.........Loss........|..........MSE........|..........SNR........|.........FWHM........|.........ShSc........|Score wto Loss|'+'\n'
        for line_idx in range(len(name_models)):
            for metric_idx,metric in enumerate(metrics_names):
                if metric_idx == 0:
                    str_aux = str_aux+ '| '+results_for_each_hop[str(int(hop))][metric][0][line_idx]+' | '+'{:.4E}'.format(results_for_each_hop[str(int(hop))][metric][1][line_idx])+' | '
                else:
                    str_aux = str_aux + results_for_each_hop[str(int(hop))][metric][0][line_idx]+' | '+'{:.4E}'.format(results_for_each_hop[str(int(hop))][metric][1][line_idx])+' | '
            if len(str(models_score_value_for_given_hop[line_idx])) == 2:
                space='   | '
            else:
                space='    | '
            str_aux=str_aux+models_scored_for_given_hop[line_idx] + ': '+str(models_score_value_for_given_hop[line_idx])+space+'\n'    
        #print average and std of models for each metric
        for metric_idx,metric in enumerate(metrics_names):
            if metric_idx == 0:
                str_aux=str_aux+'| '+'avg.'+'   | '+ '{:.4E}'.format(np.mean(results_for_each_hop[str(int(hop))][metric][1]))+' | '
            else:
                str_aux=str_aux+' avg.'+'  | '+ '{:.4E}'.format(np.mean(results_for_each_hop[str(int(hop))][metric][1]))+' | '
        str_aux=str_aux+'\n'
        for metric_idx,metric in enumerate(metrics_names):
            if metric_idx == 0:
                str_aux=str_aux+'| '+'std.'+'   | '+ '{:.4E}'.format(np.std(results_for_each_hop[str(int(hop))][metric][1]))+' | '
            else:
                str_aux=str_aux+' std.'+'  | '+ '{:.4E}'.format(np.std(results_for_each_hop[str(int(hop))][metric][1]))+' | '
        str_aux=str_aux+'\n'+'\n'
        with open(path_save_test+"extra_metrics.txt", "a") as f:
            f.write("metrics_of_each_model_for_hop_"+str(int(hop))+":\n")
            f.write(str_aux)

if show_results_for_all_hops_combined is True:
    #-----------------------------------metrics combining hops------------------------------------------------
    score_by_metric, score_by_model_combined_hops = get_scores_by_metric_and_model_for_all_hops(name_models=name_models,
                                                                                                metrics_names=metrics_names,
                                                                                                hop_size=hop_size,
                                                                                                results_for_each_hop=results_for_each_hop)

    models_scored_global, models_score_value_global = get_global_scores(name_models=name_models,
                                                                        metrics_names=['MSE','SNR','FWHM','ShSc'],
                                                                        score_by_model_combined_hops=score_by_model_combined_hops)
    #write table adding, for each model, its score using each hop value
    str_aux=''
    str_aux=str_aux+'|.....Loss.....|......MSE.....|......SNR.....|....FWHM....|.....ShSc.....|Score wto Loss|'+'\n'
    for line_idx in range(len(name_models)):
        for metric_idx,metric in enumerate(metrics_names):
            if len(str(score_by_metric[metric][1][line_idx])) == 2:
                space = '  | '
            else:
                space = ' | '
            if metric_idx == 0:
                str_aux = str_aux+'| '+score_by_metric[metric][0][line_idx]+' | '+str(score_by_metric[metric][1][line_idx])+space 
            else:
                str_aux = str_aux + score_by_metric[metric][0][line_idx]+' | '+str(score_by_metric[metric][1][line_idx])+space
        if len(str(models_score_value_global[line_idx])) == 3:
            space='  | '
        else:
            space='   | '
        str_aux=str_aux+models_scored_global[line_idx] + ': '+str(models_score_value_global[line_idx])+space+'\n'    
    str_aux=str_aux+'\n'
    with open(path_save_test+"extra_metrics.txt", "a") as f:
        f.write("metrics_for_all_hops_combined: \n")
        f.write(str_aux)


#-----------------------------------metrics visual plot------------------------------------------------
fig,ax=plt.subplots(2,2,figsize=(8,6))
for i,metric in enumerate(metrics_names[1:]):
    aux = []
    for hop_idx,hop in enumerate(hop_size):
        aux.append([])
        for model in name_models:
            aux[hop_idx].append(dict_metrics[str(int(hop))][model][metric])
    im=ax.flat[i].imshow(np.array(aux),cmap='magma')
    fig.colorbar(im, ax=ax.flat[i],fraction=0.046, pad=0.04)
    ax.flat[i].set_xticks(ticks=np.arange(len(name_models)), labels=name_models, rotation=45)
    ax.flat[i].set_yticks(ticks=np.arange(len(hop_size)), labels=[str(x) for x in hop_size] )
    ax.flat[i].set_title(metric)
plt.tight_layout()
plt.savefig(path_save_test+"metrics_visually.png")
plt.close()


#-----------------------------------visual: reconstructions------------------------------------------------
#define figures showing reconstructed spectra for each model
if len(name_models) == len(hop_size):
    fig, ax = plt.subplots(1,4,figsize=(16, 4))
    regions = [[2.8,3.2],[3.6,3.9],[1.98,2.05],[10,10.8]]
    peaks = ['GABA', 'Glx', 'NAA', 'Residuos']
    for j in range(4):
        position_sup = int(np.mean(np.argmin(np.abs(ppm_concat-regions[j][-1]),axis=1)))
        position_inf = int(np.mean(np.argmin(np.abs(ppm_concat-regions[j][0]),axis=1)))
        for i,hop in enumerate(hop_size):
            ax.flat[j].plot(np.mean(ppm_concat[:,position_sup:position_inf],axis=0), np.mean(predictions[str(int(hop))][name_models[i]][:,position_sup:position_inf],axis=0), label=name_models[i])
        ax.flat[j].plot(np.mean(ppm_concat[:,position_sup:position_inf],axis=0), np.mean(target_concat[:,position_sup:position_inf],axis=0), color='black', label='target')
        if j != 3:
            ax.flat[j].set_title('Avg. Espectros de GABA \n Pico de '+peaks[j])
        else:
            ax.flat[j].set_title('Avg. Espectros de GABA \n '+peaks[j])
        ax.flat[j].set_xlabel('Desloc. Químico (ppm)')
        ax.flat[j].set_ylabel('Espectro Normalizado')
        ax.flat[j].set_xlim(regions[j][-1],regions[j][0])
        ax.flat[j].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_save_test+"reconstructed_spects.png")
    plt.close()

#show reconstructed spectra obtained by each model for each hop value in hop_size list
if show_results_for_each_hop is True:
    for hop in hop_size:
        if all_equal(dataset_list) is True:
            fig, ax = plt.subplots(1,5,figsize=(20, 4))
            ax.flat[0].imshow(np.mean(input_spgrams[str(int(hop))][name_models[0]],axis=(0,1)),cmap='gray',vmin=-0.04,vmax=0.04,aspect='auto')
            ax.flat[0].set_title('Avg. Input para STFT com hop: '+str(int(hop)))
            initial_plot=1
        else:
            fig, ax = plt.subplots(1,4,figsize=(20, 4))
            initial_plot=0
        regions = [[2.8,3.2],[3.6,3.9],[1.98,2.05],[10,10.8]]
        peaks = ['GABA', 'Glx', 'NAA', 'Residuos']
        for j in range(4):
            position_sup = int(np.mean(np.argmin(np.abs(ppm_concat-regions[j][-1]),axis=1)))
            position_inf = int(np.mean(np.argmin(np.abs(ppm_concat-regions[j][0]),axis=1)))
            for model in name_models:
                ax.flat[j+initial_plot].plot(np.mean(ppm_concat[:,position_sup:position_inf],axis=0), np.mean(predictions[str(int(hop))][model][:,position_sup:position_inf],axis=0), label=model)
            ax.flat[j+initial_plot].plot(np.mean(ppm_concat[:,position_sup:position_inf],axis=0), np.mean(target_concat[:,position_sup:position_inf],axis=0), color='black', label='target')
            if j != 3:
                ax.flat[j+initial_plot].set_title('Avg. Espectros de GABA \n para STFT com hop: '+str(int(hop))+'\n Pico de '+peaks[j])
            else:
                ax.flat[j+initial_plot].set_title('Avg. Espectros de GABA \n para STFT com hop: '+str(int(hop))+peaks[j])
            ax.flat[j+initial_plot].set_xlabel('Desloc. Químico (ppm)')
            ax.flat[j+initial_plot].set_ylabel('Espectro Normalizado')
            ax.flat[j+initial_plot].set_xlim(regions[j][-1],regions[j][0])
            ax.flat[j+initial_plot].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path_save_test+"reconstructed_spects_from_stft_with_hop_"+str(int(hop))+".png")
        plt.close()


#-----------------------------------analysis: JensenShannon Distance------------------------------------------------
distTgt = get_models_distances_to_target(name_models=name_models,
                                         hop_size=hop_size,
                                         target_concat=target_concat,
                                         predictions=predictions)
#write table: JensenShannon distance of each model x target spectra
str_aux = ''
aux=[]
for key in list(distTgt.keys()):
    str_aux=str_aux+key+'  '+'{:.4E}'.format(distTgt[key])+ '\n'
    aux.append(distTgt[key])
str_aux=str_aux+'avg.    '+'{:.4E}'.format(np.mean(aux))+'\n'
str_aux=str_aux+'std.    '+'{:.4E}'.format(np.std(aux))+'\n'
str_aux=str_aux+'\n'
with open(path_save_test+"extra_metrics.txt", "a") as f:
        f.write("JensenShannon Distance w/ respect to target: \n")
        f.write(str_aux)

#-----------------------------------analysis: diff in estimation------------------------------------------------
#write table with difference of peak heights for each model with respect to target spectra
mean_diff_per_hop = get_mean_diff_per_hop(name_models=name_models, 
                                                    hop_size=hop_size, 
                                                    ppm_concat=ppm_concat, 
                                                    target_concat=target_concat, 
                                                    predictions=predictions)
if len(name_models) == len(hop_size):
    keys_regions = ['GABA', 'Glx', 'global']
    str_aux=''
    str_aux=str_aux+'|.........GABA........|..........Glx........|........global.......| \n'
    for line_idx in range(len(name_models)):
        for key_idx,key in enumerate(keys_regions):
            if mean_diff_per_hop[str(int(hop_size[line_idx]))][name_models[line_idx]][key] > 0: 
                space = '  |'
            else:
                space = ' |'
            if key_idx == 0:
                str_aux = str_aux + '|'+name_models[line_idx]+' | '+'{:.4E}'.format(mean_diff_per_hop[str(int(hop_size[line_idx]))][name_models[line_idx]][key])+space
            else:
                str_aux = str_aux + name_models[line_idx]+' | '+'{:.4E}'.format(mean_diff_per_hop[str(int(hop_size[line_idx]))][name_models[line_idx]][key])+space
        str_aux=str_aux+'\n'
    str_aux=str_aux+'\n'
    with open(path_save_test+"extra_metrics.txt", "a") as f:
        f.write("mean_diff_for_trained_hop: \n")
        f.write(str_aux)

#same, but considering each model in each hop value
if show_results_for_each_hop is True:
    keys_regions = ['GABA', 'Glx', 'global']
    for hop in hop_size:
        str_aux=''
        str_aux=str_aux+'|.........GABA........|..........Glx........|........global.......| \n'
        for line_idx in range(len(name_models)):
            for key_idx,key in enumerate(keys_regions):
                if mean_diff_per_hop[str(int(hop))][name_models[line_idx]][key] > 0: 
                    space = '  |'
                else:
                    space = ' |'
                if key_idx == 0:
                    str_aux = str_aux + '|'+name_models[line_idx]+' | '+'{:.4E}'.format(mean_diff_per_hop[str(int(hop))][name_models[line_idx]][key])+space
                else:
                    str_aux = str_aux + name_models[line_idx]+' | '+'{:.4E}'.format(mean_diff_per_hop[str(int(hop))][name_models[line_idx]][key])+space
            str_aux=str_aux+'\n'
        str_aux=str_aux+'\n'
        with open(path_save_test+"extra_metrics.txt", "a") as f:
            f.write("mean_diff_for_hop: "+str(int(hop))+"\n")
            f.write(str_aux)


if show_results_for_all_hops_combined is True:
    keys_regions = ['GABA', 'Glx', 'global']
    str_aux=''
    str_aux=str_aux+'|.....GABA......|.....Glx.......|....global.....| \n'
    for line_idx in range(len(name_models)):
        for key_idx,key in enumerate(keys_regions):
            counter_pos = 0
            counter_neg = 0
            for hop in hop_size:
                if mean_diff_per_hop[str(int(hop))][name_models[line_idx]][key] >= 0:
                    counter_pos=counter_pos+1
                else:
                    counter_neg=counter_neg+1
            if len(str(counter_neg))==1 and len(str(counter_pos))==1:
                space='   |'
            elif (len(str(counter_neg))==2 and len(str(counter_pos))==1) or (len(str(counter_neg))==1 and len(str(counter_pos))==2):
                space='  |'
            else:
                space=' |'
            if key_idx == 0:
                str_aux = str_aux + '|'+name_models[line_idx]+' | '+str(counter_pos)+'/'+str(counter_neg)+space
            else:
                str_aux = str_aux + name_models[line_idx]+' | '+str(counter_pos)+'/'+str(counter_neg)+space
        str_aux=str_aux+'\n'
    str_aux=str_aux+'\n'
    with open(path_save_test+"extra_metrics.txt", "a") as f:
        f.write("mean_diff_for_all_hops: \n")
        f.write(str_aux)


#-----------------------------------visual: proximity between models------------------------------------------------
#define figures with each ShapeScore, MSE, correlation and JS distance between reconstructed spectra
#obtained by different models
if len(name_models) == len(hop_size):
    proximity_shape_score, proximity_MSE, correlation_matrix, JS_distance = get_model_proximity_for_trained_hops(name_models=name_models,
                                                                                                    hop_size=hop_size,
                                                                                                    ppm_concat=ppm_concat,
                                                                                                    predictions=predictions)
    plt.figure(figsize=(16,8))
    sns.heatmap(proximity_shape_score, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma', 
                xticklabels=name_models, 
                yticklabels=name_models)
    plt.title('Shape Score: Models Reconstructions for Their Trained Hop \n Reference model: by line')
    plt.tight_layout()
    plt.savefig(path_save_test+"proximitySS_trained_hops.png")
    plt.close()

    plt.figure(figsize=(16,8))
    sns.heatmap(proximity_MSE, annot=True, annot_kws={"size": 14}, fmt='.5', cmap='magma_r', 
                xticklabels=name_models, 
                yticklabels=name_models)
    plt.title('MSE: Models Reconstructions for Their Trained Hop \n Reference model: by line')
    plt.tight_layout()
    plt.savefig(path_save_test+"proximityMSE_trained_hops.png")
    plt.close()

    plt.figure(figsize=(16,8))
    sns.heatmap(JS_distance, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma_r', 
                xticklabels=name_models, 
                yticklabels=name_models)
    plt.title('JensenShannon Distance: Models Reconstructions for Their Trained Hop \n Reference model: by line')
    plt.tight_layout()
    plt.savefig(path_save_test+"JSdist_trained_hops.png")
    plt.close()

    plt.figure(figsize=(16,8))
    sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma', 
                xticklabels=name_models, 
                yticklabels=name_models)
    plt.title('Correlation Matrix: Models Reconstructions for Their Trained Hop \n Reference model: by line')
    plt.tight_layout()
    plt.savefig(path_save_test+"correlationmatrix_trained_hops.png")
    plt.close()


if show_results_for_each_hop is True:
    for hop in hop_size:
        proximity_shape_score, proximity_MSE, correlation_matrix, JS_distance = get_model_proximity_for_given_hop(name_models=name_models,
                                                                                                    hop_value=hop,
                                                                                                    ppm_concat=ppm_concat,
                                                                                                    predictions=predictions)
        plt.figure(figsize=(16,8))
        sns.heatmap(proximity_shape_score, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma', 
                    xticklabels=name_models, 
                    yticklabels=name_models)
        plt.title('Shape Score: Models Reconstructions for STFT w/ Hop: '+str(int(hop)) +'\n Reference model: by line')
        plt.tight_layout()
        plt.savefig(path_save_test+"proximitySS_for_STFT_hop_"+str(int(hop))+".png")
        plt.close()

        plt.figure(figsize=(16,8))
        sns.heatmap(proximity_MSE, annot=True, annot_kws={"size": 14}, fmt='.5', cmap='magma_r', 
                    xticklabels=name_models, 
                    yticklabels=name_models)
        plt.title('MSE: Models Reconstructions for STFT w/ Hop: '+str(int(hop)) +'\n Reference model: by line')
        plt.tight_layout()
        plt.savefig(path_save_test+"proximityMSE_for_STFT_hop_"+str(int(hop))+".png")
        plt.close()

        plt.figure(figsize=(16,8))
        sns.heatmap(JS_distance, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma_r', 
                    xticklabels=name_models, 
                    yticklabels=name_models)
        plt.title('JensenShannon Distance: Models Reconstructions for STFT w/ Hop: '+str(int(hop)) +'\n Reference model: by line')
        plt.tight_layout()
        plt.savefig(path_save_test+"JSdist_for_STFT_hop_"+str(int(hop))+".png")
        plt.close()

        plt.figure(figsize=(16,8))
        sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 14}, fmt='.6', cmap='magma', 
                    xticklabels=name_models, 
                    yticklabels=name_models)
        plt.title('Correlation Matrix: Models Reconstructions for STFT w/ Hop: '+str(int(hop)) +'\n Reference model: by line')
        plt.tight_layout()
        plt.savefig(path_save_test+"correlationmatrix_for_STFT_hop_"+str(int(hop))+".png")
        plt.close()

print('evaluation done!')

