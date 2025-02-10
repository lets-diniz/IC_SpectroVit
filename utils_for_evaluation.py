import numpy as np
import torch
from scipy import stats
from scipy.spatial import distance


from losses import RangeMAELoss
from main_functions_adapted import run_validation
from datasets import *
from torch.utils.data import DataLoader
from metrics import calculate_shape_score, calculate_mse


def order_models(list_metric,model_names):
    """
    Function: sort models with respect to a metric. Return a list with name of models organizing
    from the one with lowest metric value to highest. 
    Inputs:
    - list_metric: list with values of a metric obtained by different models
    - model_names: list of strings with model names, in the same order as list_metric
    Outpus:
    - models_ordered: list of model names ordered from the one with lowest value of metric to the one
    with highest metric value.
    - values_order: list of metrics values organized from the lowest to the highest.
    """
    values_order = np.sort(np.array(list_metric))
    idx_values_order = np.argsort(np.array(list_metric))
    models_ordered = []
    for idx in idx_values_order:
        models_ordered.append(model_names[idx]) 
    return models_ordered, values_order

def all_equal(list_):
    """
    Function: check if all elements of a list are equal.
    Inputs:
    - list_: list containing objects to be compared
    Outputs:
    - all_elements_equal: bool, if True all elements are equal.
    """
    all_elements_equal=True
    for i in range(1,len(list_)):
        if list_[i] != list_[i-1]:
            all_elements_equal=False
    return all_elements_equal

def noise_est(ppm,x):
    """
    Function: estimate average noise in a spectrum (x) using a second order polynomial detrending. 
    Noise is estimated in region between 10 and 12 ppm.
    Inputs:
    - ppm: (N,f) array with ppm values for N different spectra
    - x: (N,f) array with N different spectra
    Outputs:
    - average noise estimation considering N spectra
    """
    noise = []
    for i in range(ppm.shape[0]):
        position_sup = np.abs(ppm[i,:]-12).argmin()
        position_inf = np.abs(ppm[i,:]-10).argmin()
        dt = np.polyfit(ppm[i,position_sup:position_inf], x[i,position_sup:position_inf], 2)
        sizeFreq = ppm[i,position_sup:position_inf].shape[0]
        stdev_Man = np.sqrt(np.sum(np.square(np.real(x[i,position_sup:position_inf] - np.polyval(dt, ppm[i,position_sup:position_inf])))) / (sizeFreq - 1))
        noise.append(stdev_Man)
    return np.mean(np.array(noise))

def get_max_gaba(ppm,x):
    """
    Function: estimate average GABA peak's height in spectrum (x).
    GABA peak is expected to be between 2.8 and 3.2 ppm. GABA peak's height is the maximum value in
    this region.
    Inputs:
    - ppm: (N,f) array with ppm values for N different spectra
    - x: (N,f) array with N different spectra
    Outputs:
    - average GABA peak height considering N spectra
    """
    max_gaba = [] 
    for i in range(ppm.shape[0]): 
        gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm[i,:] >= 2.8)), np.amin(np.where(ppm[i,:] <= 3.2))
        max_aux = x[i,gaba_min_ind:gaba_max_ind].max()
        max_gaba.append(max_aux)
    return np.mean(np.array(max_gaba))

def get_dataset(dataset):
    """
    Function: return dataset object given a name string
    Input: dataset - string with dataset name
    Output: corresponding dataset object
    """
    if dataset=='DatasetSpgramSyntheticData':
        return DatasetSpgramSyntheticData
    elif dataset=='DatasetSpgramSyntheticDataOldSTFT':
        return DatasetSpgramSyntheticDataOldSTFT
    elif dataset=='DatasetSpgramRealData':
        return DatasetSpgramRealData
    elif dataset=='DatasetSpgramRealDataOldSTFT':
        return DatasetSpgramRealDataOldSTFT
    
def get_metrics_for_different_models(path_to_test_data,list_models,name_models,hop_size,window_size,window,device,dataset_list):
    """
    Function: given different hop values and different models, get performance metrics of each model in each hop value.
    Inputs:
    - path_to_test_data: path string, path to h5 file containing transients to be used for test.
    - list_models: list with SpectroVit model instances
    - name_models: list of strings with model names in the same order as lsit_models
    - hop_size: list with hop values to be considered in the STFT applied to model input
    - window_size: int, window size to be considered in the STFT applied to model input
    - window: array with window shape with size equal to window_size
    - device: 'cpu' or 'cuda'
    - dataset_list: list of strings with datasets names indicating which dataset to be used for each model
    Outputs:
    - dict_metrics: dict with metric value obtained by each model considering each hop value for the input. Dict contains a key for 
    each hop value in hop_size. For each hop there is an inner dict with keys corresponding to model names. For each model there is an 
    inner-dict with keys corresponding to the evaluation metrics and their values for the corresponding pair (hop, model).
    """
    dict_metrics = {}
    for hop in hop_size:
        dict_metrics[str(int(hop))] = {}
        for idx,model in enumerate(list_models):
            dataset_test = get_dataset(dataset_list[idx])(path_data=path_to_test_data,
                                augment_with_noise=False,augment_with_idx_repetition=False,
                                hop_size=int(hop),window_size=window_size,window=window)
            dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False)
            loss = RangeMAELoss()
            epoch = 0
            model.eval()
            val_loss, loader_mean_mse, loader_mean_snr,loader_mean_linewidth,loader_mean_shape_score,score_challenge = run_validation(model=model, criterion=loss, loader=dataloader_test, epoch=epoch, device=device)
            dict_metrics[str(int(hop))][name_models[idx]]={'Loss': val_loss.item(), 
                                                        'MSE': loader_mean_mse.item(),
                                                        'SNR': loader_mean_snr.item(),
                                                        'FWHM':loader_mean_linewidth.item(),
                                                        'ShSc': loader_mean_shape_score.item()}
    return dict_metrics


def get_inference_instances(path_to_test_data,list_models,name_models,hop_size,window_size,window,device,dataset_list):
    """
    Function: given different hop values and different models, get samples of model predictions (reconstructed spectra)
    and their received inputs.
    Inputs:
    - path_to_test_data: path string, path to h5 file containing transients to be used for test.
    - list_models: list with SpectroVit model instances
    - name_models: list of strings with model names in the same order as list_models
    - hop_size: list with hop values to be considered in the STFT applied to model input
    - window_size: int, window size to be considered in the STFT applied to model input
    - window: array with window shape with size equal to window_size
    - device: 'cpu' or 'cuda'
    - dataset_list: list of strings with datasets names indicating which dataset to be used for each model
    Outputs:
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in test data.
    - target_concat: array containing the target spectrum for each transient in test data.
    - ppm_concat: array containing the ppm values for the spectrum for each transient in test data.
    - input_spgrams: dict containing the input for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the input obtained for each transient in test data considering a given hop.
    """
    predictions = {}
    input_spgrams = {}
    with torch.no_grad():
        for hop_idx,hop in enumerate(hop_size):
            predictions[str(int(hop))] = {}
            input_spgrams[str(int(hop))] = {}
            for i,model in enumerate(list_models):
                dataset_test = get_dataset(dataset_list[i])(path_data=path_to_test_data,
                                augment_with_noise=False,augment_with_idx_repetition=False,
                                hop_size=int(hop),window_size=window_size,window=window)
                dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False)
                first_batch = True
                count = 0
                if hop_idx == 0:
                    ppm_concat = np.empty((len(dataset_test),2048))
                    target_concat = np.empty((len(dataset_test),2048))

                for data in dataloader_test:
                    input_, target, ppm = data[0], data[1], data[2]
                    input_ = input_.to(device)
                    target = target.numpy()
                    ppm = ppm.numpy()
                    if hop_idx == 0:
                        ppm_concat[count*ppm.shape[0]:(count+1)*ppm.shape[0],:] = ppm
                        target_concat[count*target.shape[0]:(count+1)*target.shape[0],:] = target
                    prediction = model(input_).cpu().detach().numpy()
                    input_spgram = input_.cpu().detach().numpy()
                    if first_batch == True:
                        predictions[str(int(hop))][name_models[i]] = prediction
                        input_spgrams[str(int(hop))][name_models[i]] = input_spgram
                        first_batch=False
                    else:
                        predictions[str(int(hop))][name_models[i]] = np.concatenate((predictions[str(int(hop))][name_models[i]],prediction),axis=0)
                        input_spgrams[str(int(hop))][name_models[i]] = np.concatenate((input_spgrams[str(int(hop))][name_models[i]],input_spgram),axis=0)
                    count=count+1

    return predictions, target_concat, ppm_concat,input_spgrams


def get_stats_from_predictions(name_models,hop_size,target_concat,predictions):
    """
    Function: get statistics from a set of predictions of different models for different hop values.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values considered to generate predictions
    - target_concat: array containing the target spectrum for each transient in a test data
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    - tgt_stats: dict with keys of stats of target spectra
    - pred_stats: dict with keys of desired stats of models' predictions. For each stat there is a list,
    if there is one hop value for each model, the list contains the average stat value for the model predictions
    for its corresponding hop value. If  there isn't a match between hops and models, for every hop (in the order of
    hop_size) we get the average stat value for every model prediction considering the given hop (in the order
    of name_models).
    """
    pred_stats = {'mean':[],'std':[],'median':[],'skew':[],'kurtosis':[]}
    tgt_stats = {}
    tgt_stats['mean']= np.mean(target_concat.flatten())
    tgt_stats['std']= np.std(target_concat.flatten())
    tgt_stats['median']= np.median(target_concat.flatten())
    tgt_stats['skew']= stats.skew(target_concat.flatten())
    tgt_stats['kurtosis']= stats.kurtosis(target_concat.flatten())
    for i,hop in enumerate(hop_size):
        if len(name_models) == len(hop_size):
            pred_stats['mean'].append(np.mean(predictions[str(int(hop))][name_models[i]].flatten()))
            pred_stats['std'].append(np.std(predictions[str(int(hop))][name_models[i]].flatten()))
            pred_stats['median'].append(np.median(predictions[str(int(hop))][name_models[i]].flatten()))
            pred_stats['skew'].append(stats.skew(predictions[str(int(hop))][name_models[i]].flatten()))
            pred_stats['kurtosis'].append(stats.kurtosis(predictions[str(int(hop))][name_models[i]].flatten()))
        else:
            for name in name_models:
                pred_stats['mean'].append(np.mean(predictions[str(int(hop))][name].flatten()))
                pred_stats['std'].append(np.std(predictions[str(int(hop))][name].flatten()))
                pred_stats['median'].append(np.median(predictions[str(int(hop))][name].flatten()))
                pred_stats['skew'].append(stats.skew(predictions[str(int(hop))][name].flatten()))
                pred_stats['kurtosis'].append(stats.kurtosis(predictions[str(int(hop))][name].flatten()))
    return tgt_stats, pred_stats

def get_performance_metric_for_trained_hop(name_models,hop_size,dict_metrics,metrics_names):
    """
    Function: given a set of models and their corresponding hop values get their performance with respect
    to a set of metrics.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values (one hop for each model)
    - dict_metrics: dict with metric value obtained by each model considering each hop value for the input. Dict contains a key for 
    each hop value in hop_size. For each hop there is an inner dict with keys corresponding to model names. For each model there is an 
    inner-dict with keys corresponding to the evaluation metrics and their values for the corresponding pair (hop, model).
    - metrics_names: list of strings containing the metrics to be considered
    Outputs:
    - results_for_their_right_hop: dict with models ordered by their performance on different metrics considering
    results obtained by the model when the input uses the expect hop value. Dict contains a key for each metric in metrics_names.
    For each metric there is a list with two inner lists, the first one contains the models names ordered from
    best to worse performance in that metric, and the second one the corresponding metric value for the ordered models.
    """
    results_for_their_right_hop = {}
    for metric in metrics_names:
            results_for_their_right_hop[metric] = [[],[]]
            aux = []
            for idx,model in enumerate(name_models):
                aux.append(dict_metrics[str(int(hop_size[idx]))][model][metric])
            aux_model, aux_value = order_models(list_metric=aux,model_names=name_models)
            if metric == 'SNR' or metric == 'ShSc':
                #from higher to lower
                results_for_their_right_hop[metric][0] = aux_model[::-1]
                results_for_their_right_hop[metric][1] = np.flip(aux_value)
            else:
                #from lower to higher
                results_for_their_right_hop[metric][0] = aux_model
                results_for_their_right_hop[metric][1] = aux_value
    return results_for_their_right_hop

def get_scores_for_given_hop(name_models,metrics_names,results_for_given_hop):
    """
    Function: given a set of models and their corresponding hop values score their performance on
    a set of metrics. Give scores depending on their position with respect to other models for each metric and
    then add their scores for each metric, to have a total score.
    If a model is 1º in a metric and 3º in another, its score is 4.
    If for a certain metric two or more models have the same value performance, we add the same value for their score.
    Inputs:
    - name_models: list of strings with model names
    - metrics_names: list of strings containing the metrics to be considered
    - results_for_their_right_hop: dict with models ordered by their performance on different metrics considering
    results obtained by the model when the input uses the expect hop value. Dict contains a key for each metric in metrics_names.
    For each metric there is a list with two inner lists, the first one contains the models names ordered from
    best to worse performance in that metric, and the second one the corresponding metric value for the ordered models.
    Outputs:
    - models_scored_for_given_hop: list with ordered models from best to worse total score.
    - models_score_value_for_given_hop: list with scores of each model.
    """
    score_model_for_given_hop = {}
    for model in name_models:
        score_model_for_given_hop[model] = 0
    for metric in metrics_names:
        no_repeat = np.unique(results_for_given_hop[metric][1])
        max_point = len(no_repeat)
        for model_idx in range(len(results_for_given_hop[metric][0])):
            if model_idx == 0:
                add_to_score = max_point
            else:
                if results_for_given_hop[metric][1][model_idx] != results_for_given_hop[metric][1][model_idx-1]:
                    add_to_score = add_to_score-1
            score_model_for_given_hop[results_for_given_hop[metric][0][model_idx]] = score_model_for_given_hop[results_for_given_hop[metric][0][model_idx]] + add_to_score
    aux = []
    for model in name_models:
        aux.append(score_model_for_given_hop[model])
    models_scored_for_given_hop, models_score_value_for_given_hop = order_models(list_metric=aux,model_names=name_models)
    return models_scored_for_given_hop[::-1], models_score_value_for_given_hop[::-1]

def get_results_for_each_hop(name_models,hop_size,metrics_names,dict_metrics):
    """
    Function: given a set of models and a set of hop values, get models performance with respect
    to a set of metrics for each hop value.
    Similar to get_performance_metric_for_trained_hop, but instead of considering model performance
    in only one hop, consider model performance for each possible hop.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values
    - metrics_names: list of strings containing the metrics to be considered
    - dict_metrics: dict with metric value obtained by each model considering each hop value for the input. Dict contains a key for 
    each hop value in hop_size. For each hop there is an inner dict with keys corresponding to model names. For each model there is an 
    inner-dict with keys corresponding to the evaluation metrics and their values for the corresponding pair (hop, model).
    Outputs:
    - results_for_each_hop: dict with models ordered by their performance on different metrics for each hop value. Dict contains a key for 
    each hop in hop_size. For each hop there is an inner dict with keys being the metrics in metrics_names.
    For each metric there is a list with two inner lists, the first one contains the models names ordered from
    best to worse performance in that metric (and that hop), and the second one the corresponding metric value for the ordered models.
    """
    results_for_each_hop = {}
    for hop in hop_size:
        results_for_each_hop[str(int(hop))] = {}
        for metric in metrics_names:
            results_for_each_hop[str(int(hop))][metric] = [[],[]]
            aux = []
            for model in name_models:
                aux.append(dict_metrics[str(int(hop))][model][metric])
            aux_model,aux_value = order_models(list_metric=aux,model_names=name_models)
            if metric == 'SNR' or metric == 'ShSc':
                results_for_each_hop[str(int(hop))][metric][0] = aux_model[::-1]
                results_for_each_hop[str(int(hop))][metric][1] = np.flip(aux_value)
            else:
                results_for_each_hop[str(int(hop))][metric][0] = aux_model
                results_for_each_hop[str(int(hop))][metric][1] = aux_value
    return results_for_each_hop

def get_scores_by_metric_and_model_for_all_hops(name_models,metrics_names,hop_size,results_for_each_hop):
    """
    Function: given a set of models and a set of hop values score their performance on
    a set of metrics. Considering all hop values, give scores depending on model position with respect 
    to other models for each metric. We add scores for all hop values for each metric, so we have the models
    scored by metric.
    If, for a certain metric, a model is 1º in a hop value and 3º in another, its score is 4.
    If, for a certain metric, two or more models have the same value performance for the same hop, we
    add the same value for their score.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values
    - metrics_names: list of strings containing the metrics to be considered
    - results_for_each_hop: dict with models ordered by their performance on different metrics for each hop value. Dict contains a key for 
    each hop in hop_size. For each hop there is an inner dict with keys being the metrics in metrics_names.
    For each metric there is a list with two inner lists, the first one contains the models names ordered from
    best to worse performance in that metric (and that hop), and the second one the corresponding metric value for the ordered models.
    Outputs:
    - score_by_metric: dict with models ordered by their performance in each metric considering all hop values. Dict contains as keys the metrics
    in metrics_names. For each metric, there is a list containing two lists. The first one with the models ordered from best to
    worse on the metric. And the second one their scores.
    - score_by_model_combined_hops: dict global model score in each metric by considering all hops. Dict contains keys as 
    the model names. For every model there is a dict with metrics as keys. In this inner dict, there is
    the global score of the model in a given metric. (Same info as score_by_metric but ordered differently).
    """
    score_by_model_combined_hops = {}
    for name in name_models:
        score_by_model_combined_hops[name] = {}
        for metric_name in metrics_names:
            score_by_model_combined_hops[name][metric_name] = 0
    for metric in metrics_names:
        for hop in hop_size:
            no_repeat = np.unique(results_for_each_hop[str(int(hop))][metric][1])
            max_point = len(no_repeat)
            for model_idx in range(len(results_for_each_hop[str(int(hop))][metric][0])):
                model_ref = results_for_each_hop[str(int(hop))][metric][0][model_idx]
                if model_idx == 0:
                    add_to_score = max_point
                else:
                    if results_for_each_hop[str(int(hop))][metric][1][model_idx] != results_for_each_hop[str(int(hop))][metric][1][model_idx-1]: 
                        add_to_score = add_to_score-1                     
                score_by_model_combined_hops[model_ref][metric] = score_by_model_combined_hops[model_ref][metric] + add_to_score
    
    score_by_metric = {}
    for metric in metrics_names:
        aux = []
        for model in name_models:
            aux.append(score_by_model_combined_hops[model][metric])
        models_ordered_for_metric, models_values_for_metric = order_models(list_metric=aux,model_names=name_models)
        score_by_metric[metric] = [models_ordered_for_metric[::-1],models_values_for_metric[::-1]]
    return score_by_metric, score_by_model_combined_hops

def get_global_scores(name_models,metrics_names,score_by_model_combined_hops):
    """
    Function: given a set of models and their global performance in a set of metrics, we combine
    their performances for every metric to get a total model score.
    If a model has score 5 in a metric and 7 in another, its total score is 12.
    Inputs:
    - name_models: list of strings with model names
    - metrics_names: list of strings containing the metrics to be considered
    - score_by_model_combined_hops: dict global model score in each metric by considering all hops. Dict contains keys as 
    the model names. For every model there is a dict with metrics as keys. In this inner dict, there is
    the global score of the model in a given metric.
    Outputs:
    - model_scored_global_metrics: list with models names ordered from global best to global worst
    - model_score_value_global_metrics: list with models scores ordered from best to worst
    """
    total_score_models = []
    for model in name_models:
        aux=0
        for metric in metrics_names:
            aux=aux+score_by_model_combined_hops[model][metric]
        total_score_models.append(aux)
    model_scored_global_metrics,model_score_value_global_metrics = order_models(list_metric=total_score_models,model_names=name_models)
    return model_scored_global_metrics[::-1], model_score_value_global_metrics[::-1]

def get_models_mean_diff_for_given_hop(name_models, hop_value, ppm_concat, target_concat, predictions):
    """
    Function: For a set of models and a given hop value, get the average difference between model predicted spectrum
    and target spectrum for different spectral regions: globally, GABA peak and Glx peak.
    Inputs:
    - name_models: list of strings with model names
    - hop_value: int, hop value to be considered
    - ppm_concat: array containing the ppm values for the spectrum for each transient in a test data.
    - target_concat: array containing the target spectrum for each transient in a test data.
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    - height_diff_per_model: dict with average difference between prediction and target for every model considering
    predictions obtained for an input generated by using the given hop value in the STFT. Dict contains keys as the models names,
    for each model there is an inner dict with regions names: GABA, Glx and global. For every spectral region,
    there is the average difference between the model prediction value and target value in the region.
    """
    height_diff_per_model = {}
    regions = [[2.8,3.2],[3.6,3.9]]
    name_regions = ['GABA', 'Glx']
    for model in name_models:
        height_diff_per_model[model] = {}
        for j in range(2):
            aux = []
            for q in range(ppm_concat.shape[0]):
                position_sup = np.abs(ppm_concat[q,:]-regions[j][-1]).argmin()
                position_inf = np.abs(ppm_concat[q,:]-regions[j][0]).argmin()
                diff = np.abs(predictions[str(int(hop_value))][model][q,position_sup:position_inf])-np.abs(target_concat[q,position_sup:position_inf])
                aux.append(diff)
            height_diff_per_model[model][name_regions[j]] = np.mean(np.array(aux))
        aux=[]
        for q in range(ppm_concat.shape[0]):
            diff = np.abs(predictions[str(int(hop_value))][model][q,:])-np.abs(target_concat[q,:])
            aux.append(diff)
        height_diff_per_model[model]['global'] = np.mean(np.array(aux))
    return height_diff_per_model

def get_mean_diff_per_hop(name_models, hop_size, ppm_concat, target_concat, predictions):
    """
    Function: For a set of models and a set of hop values, get the average difference between model predicted spectrum
    and target spectrum for different spectral regions: globally, GABA peak and Glx peak.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values
    - ppm_concat: array containing the ppm values for the spectrum for each transient in a test data.
    - target_concat: array containing the target spectrum for each transient in a test data.
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    - mean_diff_per_hop: dict with average difference between prediction and target for every model and every hop value. 
    Dict contain a key for every hop in hop_size. For every hop there is an inner dict with keys as the models names.
    For each model there is an inner dict with regions names: GABA, Glx and global. For every spectral region,
    there is the average difference between the model prediction value and target value in the region.
    """
    mean_diff_per_hop = {}
    for hop in hop_size:
        mean_diff_per_model = get_models_mean_diff_for_given_hop(name_models=name_models, 
                                                                           hop_value=hop, 
                                                                           ppm_concat=ppm_concat, 
                                                                           target_concat=target_concat, 
                                                                           predictions=predictions)
        mean_diff_per_hop[str(int(hop))] = mean_diff_per_model
    return mean_diff_per_hop


def get_model_proximity_for_trained_hops(name_models,hop_size,ppm_concat,predictions):
    """
    Function: given a set of models and their corresponding hop values get the proximity between
    models predictions considering MSE, shape score, JS distance and correlation as metrics
    for prediction proximity.
    to a set of metrics.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values (one hop for each model)
    - ppm_concat: array containing the ppm values for the spectrum for each transient in a test data.
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    If there is one hop for every model:
    - proximity_shape_score: Shape Score matrix between models predictions considering inputs obtained
    using their corresponding hop value. In other words, in position (0,1), one will find the Shape Score between predictions
    of model 0 and model 1 considering that model 0 predictions were obtained from inputs that were created using
    hop idx 0, and that model 1 predictions were obtained from inputs that were created using
    hop idx 1. Target for shape score is line oriented, i.e., in the above example, Shape Score is calculated
    considering model 0 prediction as target.
    - proximity_MSE: MSE matrix between models predictions considering inputs obtained
    using their corresponding hop value. In other words, in position (0,1), one will find the MSE between predictions
    of model 0 and model 1 considering that model 0 predictions were obtained from inputs that were created using
    hop idx 0, and that model 1 predictions were obtained from inputs that were created using
    hop idx 1. Target for MSE is line oriented, i.e., in the above example, MSE is calculated
    considering model 0 prediction as target.
    - correlation_matrix: correlation  matrix between models predictions considering inputs obtained
    using their corresponding hop value. In other words, in position (0,1), one will find the correlation between predictions
    of model 0 and model 1 considering that model 0 predictions were obtained from inputs that were created using
    hop idx 0, and that model 1 predictions were obtained from inputs that were created using
    hop idx 1. 
    - JS_distance: Jensen Shannon distance matrix between models predictions considering inputs obtained
    using their corresponding hop value. In other words, in position (0,1), one will find the JS distance between predictions
    of model 0 and model 1 considering that model 0 predictions were obtained from inputs that were created using
    hop idx 0, and that model 1 predictions were obtained from inputs that were created using
    hop idx 1.
    Else:
    None
    """
    if len(hop_size) == len(name_models):
        proximity_shape_score = np.empty((len(name_models),len(name_models)))
        proximity_MSE = np.empty((len(name_models),len(name_models)))
        JS_distance =  np.empty((len(name_models),len(name_models)))
        for model_idx,model in enumerate(name_models):
            aux_tgt = predictions[str(int(hop_size[model_idx]))][model]
            for model_idx_aux,model_aux in enumerate(name_models):
                if model_idx_aux != model_idx:
                    aux = predictions[str(int(hop_size[model_idx_aux]))][model_aux]
                    aux_ss = []
                    aux_mse = []
                    for q in range(ppm_concat.shape[0]):
                        aux_ss.append(calculate_shape_score(x=aux[q,:], y=aux_tgt[q,:],ppm=ppm_concat[q,:]))
                        aux_mse.append(calculate_mse(x=aux[q,:], y=aux_tgt[q,:], ppm=ppm_concat[q,:]))
                    proximity_shape_score[model_idx,model_idx_aux] = np.mean(np.array(aux_ss))
                    proximity_MSE[model_idx,model_idx_aux] = np.mean(np.array(aux_mse))
                else:
                    proximity_shape_score[model_idx,model_idx_aux] = 1
                    proximity_MSE[model_idx,model_idx_aux] = 0
                aux = predictions[str(int(hop_size[model_idx_aux]))][model_aux]
                hist_tgt, bins_edges = np.histogram(aux_tgt.flatten(),bins=100)
                hist_aux, bins_edges = np.histogram(aux.flatten(),bins=100)
                JS_distance[model_idx,model_idx_aux] = distance.jensenshannon(hist_tgt/hist_tgt.sum(),hist_aux/hist_aux.sum())
        
        for i,hop in enumerate(hop_size):
            if i == 0:
                data = predictions[str(int(hop))][name_models[i]].flatten()
            else:
                data = np.vstack([data,predictions[str(int(hop))][name_models[i]].flatten()])
        correlation_matrix = np.corrcoef(data)
        return proximity_shape_score, proximity_MSE, correlation_matrix, JS_distance
    else:
        print("Can't get similarity matrix with nmr of hops different from nmr of models")
        return None, None, None, None


def get_model_proximity_for_given_hop(name_models,hop_value,ppm_concat,predictions):
    """
    Function: given a set of models and a given hop value get the proximity between
    models predictions considering MSE, shape score, JS distance and correlation as metrics
    for prediction proximity.
    to a set of metrics.
    Inputs:
    - name_models: list of strings with model names
    - hop_value: int, desired hop value
    - ppm_concat: array containing the ppm values for the spectrum for each transient in a test data.
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    - proximity_shape_score: Shape Score matrix between models predictions considering inputs obtained
    using the given hop value. In other words, in position (0,1), one will find the Shape Score between predictions
    of model 0 and model 1 considering that both models predictions were obtained from inputs that were created using
    hop=hop_value. Target for shape score is line oriented, i.e., in the above example, Shape Score is calculated
    considering model 0 prediction as target.
    - proximity_MSE: MSE matrix between models predictions considering inputs obtained
    using the given hop value. In other words, in position (0,1), one will find the MSE between predictions
    of model 0 and model 1 considering that both models predictions were obtained from inputs that were created using
    hop=hop_value. Target for MSE is line oriented, i.e., in the above example, MSE is calculated
    considering model 0 prediction as target.
    - correlation_matrix: correlation matrix between models predictions considering inputs obtained
    using the given hop value. In other words, in position (0,1), one will find the correlation between predictions
    of model 0 and model 1 considering that both models predictions were obtained from inputs that were created using
    hop=hop_value. 
    - JS_distance: Jensen Shannon distance matrix between models predictions considering inputs obtained
    using the given hop value. In other words, in position (0,1), one will find the JS distance between predictions
    of model 0 and model 1 considering that both models predictions were obtained from inputs that were created using
    hop=hop_value. 
    """
    proximity_shape_score = np.empty((len(name_models),len(name_models)))
    proximity_MSE = np.empty((len(name_models),len(name_models)))
    JS_distance = np.empty((len(name_models),len(name_models)))
    for model_idx,model in enumerate(name_models):
        aux_tgt = predictions[str(int(hop_value))][model]
        for model_idx_aux,model_aux in enumerate(name_models):
            if model_idx_aux != model_idx:
                aux = predictions[str(int(hop_value))][model_aux]
                aux_ss = []
                aux_mse = []
                for q in range(ppm_concat.shape[0]):
                    aux_ss.append(calculate_shape_score(x=aux[q,:], y=aux_tgt[q,:],ppm=ppm_concat[q,:]))
                    aux_mse.append(calculate_mse(x=aux[q,:], y=aux_tgt[q,:], ppm=ppm_concat[q,:]))
                proximity_shape_score[model_idx,model_idx_aux] = np.mean(np.array(aux_ss))
                proximity_MSE[model_idx,model_idx_aux] = np.mean(np.array(aux_mse))
            else:
                proximity_shape_score[model_idx,model_idx_aux] = 1
                proximity_MSE[model_idx,model_idx_aux] = 0
            aux = predictions[str(int(hop_value))][model_aux]
            hist_tgt, bins_edges = np.histogram(aux_tgt.flatten(),bins=100)
            hist_aux, bins_edges = np.histogram(aux.flatten(),bins=100)
            JS_distance[model_idx,model_idx_aux] = distance.jensenshannon(hist_tgt/hist_tgt.sum(),hist_aux/hist_aux.sum())
    
    for i,model in enumerate(name_models):
        if i == 0:
            data = predictions[str(int(hop_value))][model].flatten()
        else:
            data = np.vstack([data,predictions[str(int(hop_value))][model].flatten()])
    correlation_matrix = np.corrcoef(data)
    return proximity_shape_score, proximity_MSE, correlation_matrix, JS_distance


def get_models_distances_to_target(name_models,hop_size,target_concat,predictions):
    """
    Function: For a set of models and a set of hop values, get the average JensenShannon distance between
    model predictions and target spectra.
    Inputs:
    - name_models: list of strings with model names
    - hop_size: list with hop values
    - target_concat: array containing the target spectrum for each transient in a test data.
    - predictions: dict containing the predictions for each model given each hop. Dict contains a  key for each hop value in hop_size.
    For each hop there is an inner dict with keys corresponding to model names. For each model there is an array containing
    the predictions obtained for each transient in a test data.
    Outputs:
    - distTgt: dict with JS distance between prediction and target for every model. 
    If there is one hop value for each model, dict has as keys models names and the distance is only 
    evaluated for predictions obtained by the model when the input was generated using its corresponding hop value.
    If there is not one hop for every model, then we evaluate the distance of every model for every hop.
    In this case, dict contains as keys the model name and hop value and the JS distance between prediction
    and target considering predicitons obtained by the pair (hop,model).
    """
    distTgt = {}
    tgt_hist, bin_edges = np.histogram(target_concat.flatten(),bins=100)
    tgt_norm = tgt_hist/tgt_hist.sum()
    if len(name_models) == len(hop_size):
        for hop_idx,hop in enumerate(hop_size):
            hist, bin_edges = np.histogram(predictions[str(int(hop))][name_models[hop_idx]].flatten(),bins=100)
            hist_norm=hist/hist.sum()
            distTgt[name_models[hop_idx]] = distance.jensenshannon(tgt_norm,hist_norm)
    else:
        for hop_idx,hop in enumerate(hop_size):
            for name_idx in range(len(name_models)):
                hist, bin_edges = np.histogram(predictions[str(int(hop))][name_models[name_idx]].flatten(),bins=100)
                hist_norm=hist/hist.sum()
                distTgt[name_models[name_idx]+' and hop '+str(int(hop))] = distance.jensenshannon(tgt_norm,hist_norm)
    return distTgt
    

