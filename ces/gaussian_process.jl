import Pkg; Pkg.activate(".")
import JLD2
import YAML
import ClimaCalibrate as CAL
using ScikitLearn
using LinearAlgebra
using Statistics

import CalibrateEmulateSample as CES 
using CalibrateEmulateSample.Emulators 
using EnsembleKalmanProcesses.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.MarkovChainMonteCarlo
import EnsembleKalmanProcesses as EKP


gppackage = Emulators.SKLJL()


ekp_fpath = "ces/eki_file.jld2"
eki_obj = JLD2.load_object(ekp_fpath);
prior = CAL.get_prior("ces/prior_prognostic_pi_entr_smooth_entr_detr_impl_0M_v1.toml")

train_iopairs = CES.Utilities.get_training_points(eki_obj, 1:7)
test_iopairs = CES.Utilities.get_training_points(eki_obj, 8:8)

index_max = 180

function clean_iopairs_nans(iopairs::PairedDataContainer, index_max)
    inputs = get_inputs(iopairs)
    outputs = get_outputs(iopairs)
    
    n_samples_in = size(inputs, 2)
    
    # Identify columns in the output matrix that contain any NaNs.
    # `any` over `dims=1` checks each column. `vec` converts the resulting row matrix to a vector.
    nan_output_cols = vec(any(isnan.(outputs), dims=1))
    
    # Indices of columns to keep (where there are no NaNs)
    kept_indices = findall(.!nan_output_cols)
    
    # Filter inputs and outputs to retain only the "good" columns
    cleaned_inputs = inputs[:, kept_indices]
    cleaned_outputs = outputs[1:index_max, kept_indices]
    
    n_removed = n_samples_in - length(kept_indices)
    println("Removed $(n_removed) samples with NaNs out of $(n_samples_in) total.")
    
    return PairedDataContainer(cleaned_inputs, cleaned_outputs)
end

cleaned_train_iopairs = clean_iopairs_nans(train_iopairs, index_max)
cleaned_test_iopairs = clean_iopairs_nans(test_iopairs, index_max)

# # Get all diagonal entries from each observation's covariance matrix
# diag_entries = vcat([diag(cov) for cov in eki_obj.observation_series.observations[1].covs[:]])

# # Create diagonal matrix from these entries
# full_cov_matrix = Diagonal(vcat([diag(hcat(obs.covs...)) for obs in eki_obj.observation_series.observations]...))
# # full_cov_matrix = Diagonal(diag(eki_obj.observation_series.observations[1].covs[1]))

# gauss_proc = Emulators.GaussianProcess(gppackage, noise_learn = true) 



# optimize_hyperparameters!(emulator_gp)


full_cov_matrix = Diagonal(vcat([diag(hcat(obs.covs...)) for obs in eki_obj.observation_series.observations[1:2]]...))
# full_cov_matrix = Diagonal(diag(eki_obj.observation_series.observations[1].covs[1]))


# full_cov_matrix = cov(get_outputs(cleaned_iopairs))

nugget = 1e-3
overrides = Dict(
    "verbose" => true,
    "scheduler" => DataMisfitController(terminate_at = 20.0),
    "cov_sample_multiplier" => 1.0,
    "n_iteration" => 5,
    "n_features_opt" => 40,
    "n_ensemble" => 40, #tune 
)
# add more data
n_features = 100
n_params = 21
kernel_structure = SeparableKernel(LowRankFactor(1, nugget), OneDimFactor()) # tune
mlt = ScalarRandomFeatureInterface(
    n_features,
    n_params,
    kernel_structure = kernel_structure,
    optimizer_options = overrides,
)


emulator_gp = Emulator(mlt, 
                    cleaned_train_iopairs;
                    obs_noise_cov = full_cov_matrix,
                    normalize_inputs = true,
                    retained_svd_frac = 0.95)


optimize_hyperparameters!(emulator_rf)
