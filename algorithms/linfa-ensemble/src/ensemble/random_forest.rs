use super::{EnsembleLearner, EnsembleLearnerParams};
use ndarray::{Array1, Array, Array2};
use linfa::{Dataset};
use linfa::prelude::{Predict, PredictInplace, Fit};
use linfa_trees::{DecisionTree, DecisionTreeParams};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand::rngs::SmallRng;


pub type RandomForestParams<F, L> = EnsembleLearnerParams<F, DecisionTreeParams<F, L>>;
pub type RandomForest<F, L> = EnsembleLearner<F, L, DecisionTree<F, L>, Array2<L>>;

pub fn rf_test() {

    let num_samples = 3;
    let num_features = 5;
    let mut rng = SmallRng::seed_from_u64(42);

    println!("Creating Data");
    let data = Array::random_using((num_samples, num_features), Uniform::new(-1., 1.), &mut rng);
    let targets = (0..num_samples).collect::<Array1<usize>>();
    let dataset = Dataset::new(data, targets);

    println!("Fitting Tree as baseline");
    let tree_params: DecisionTreeParams<f64, usize> = DecisionTree::params();
    let tree_model = tree_params.fit(&dataset).unwrap();


    println!("Creating Ensemble");
    let dt: DecisionTreeParams<f64, usize> = DecisionTree::params();
    let mut learner: RandomForestParams<f64, usize> = EnsembleLearnerParams::new();
    learner.ensemble_size(5).bootstrap_proportion(0.6).model_params(dt);

    println!("Fitting Ensemble Model");
    let model = learner.fit(&dataset).unwrap();


    println!("Creating Prediction Data");
    let data = Array::random_using((num_samples, num_features), Uniform::new(-1., 1.), &mut rng);
    println!("Predicting");
    let predictions_tree: Array1<usize> = tree_model.predict(&data);

    let predictions_ensemble = model.generate_predictions(&data);
    let ranked_predictions_ensemble = model.aggregate_predictions(predictions_ensemble);

    //println!("Predictions: \n{:?}", predictions_ensemble);
    println!("Ranked Predictions: \n{:?}", ranked_predictions_ensemble);

    let mut y_array = model.default_target(&dataset.records);


    let final_predictions_ensemble = model.predict(&dataset);
    println!("Final Predictions: \n{:?}", final_predictions_ensemble);
}
