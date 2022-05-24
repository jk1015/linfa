use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned},
    error::{Error},
    traits::*,
    DatasetBase,
};
use ndarray::{
    Array1, Array2, ArrayBase, Axis,
    Data, Ix2,
};
use std::marker::PhantomData;
use std::collections::HashMap;

// Add a wrapper function for getting stats out of predictors
// Want to remove T here, but it causes unconstrained parameter error in impl block below
pub struct EnsembleLearner<F, E, M, T> {
    pub models: Vec<M>,
    _t: PhantomData<(F, E, T)>
}

impl <'b, F: Clone, E, T, M: PredictInplace<Array2<F>, T>> EnsembleLearner<F, E, M, T>
where
    E: Copy + std::cmp::Eq + std::hash::Hash + 'b,
    T: AsTargets<Elem = E>,
{

    pub fn generate_predictions(&'b self, x: &'b Array2<F>) -> impl Iterator<Item = T> + 'b {
        self.models.iter().map(move |m| {
            let result = m.predict(x);
            result
        })
    }


    // Consumes prediction iterator to return all predictions made by any model
    // Orders predictions by total number of models giving that prediciton
    pub fn aggregate_predictions(&self, ys: impl Iterator<Item=T> + 'b)
    -> Array1<Vec<(Array1<E>, usize)>>
    {
        let mut prediction_maps = Vec::new();

        for y in ys {
            let targets = y.as_multi_targets();
            let no_targets = targets.shape()[0];

            for i in 0..no_targets {
                if prediction_maps.len() == i {
                    prediction_maps.push(HashMap::new())
                }
                //Still need to take ownership here to get data out of view
                //Might be better to store all predictions elsewhere and return a view on them here?
                *prediction_maps[i].entry(y.as_multi_targets().index_axis(Axis(0), i).to_owned()).or_insert(0) += 1;
            }

        }

        let mut prediction_array = Array1::from_elem(prediction_maps.len(), Vec::new());

        for i in 0..prediction_maps.len() {
            //I think these need to copy data again and would also be nicer if they contained a view?
            prediction_array[i] = prediction_maps[i].to_owned().into_iter().collect();
            prediction_array[i].sort_by(|(_, a), (_, b)| b.cmp(a))

        }


        prediction_array

    }

}

impl <F: Clone, E, T, M: PredictInplace<Array2<F>, T>>
PredictInplace<Array2<F>, T> for EnsembleLearner<F, E, M, T>
where
    E: Copy + std::cmp::Eq + std::hash::Hash,
    T: AsTargets<Elem = E> + AsTargetsMut<Elem = E>,
{
    fn predict_inplace(&self, x: &Array2<F>, y: &mut T) {
        let mut y_array = y.as_multi_targets_mut();
        assert_eq!(
            x.nrows(),
            y_array.len(),
            "The number of data points must match the number of output targets."
        );

        let mut predictions = self.generate_predictions(x);
        let aggregated_predictions = self.aggregate_predictions(&mut predictions);

        for (target, output) in y_array.axis_iter_mut(Axis(0)).zip(aggregated_predictions.into_iter()) {
            for (t, o) in target.into_iter().zip(output[0].0.iter()) {
                *t = *o;
            }
        }
    }

    fn default_target(&self, x: &Array2<F>) -> T{
        self.models[0].default_target(x)
    }
}

pub struct EnsembleLearnerParams<F, E, P> {
    pub ensemble_size: usize,
    pub bootstrap_proportion: f64,
    pub model_params: Option<P>,
    _t: PhantomData<(F, E)>

}

impl<F, E, P> EnsembleLearnerParams<F, E, P> {
    pub fn new() -> EnsembleLearnerParams<F, E, P> {
        EnsembleLearnerParams {ensemble_size: 1, bootstrap_proportion: 1.0, model_params: None, _t: PhantomData}
    }

    pub fn ensemble_size(&mut self, size: usize) -> &mut EnsembleLearnerParams<F, E, P> {
        self.ensemble_size = size;
        self
    }

    pub fn bootstrap_proportion(&mut self, proportion: f64) -> &mut EnsembleLearnerParams<F, E, P> {
        self.bootstrap_proportion = proportion;
        self
    }

    pub fn model_params(&mut self, params: P) -> &mut EnsembleLearnerParams<F, E, P>  {
        self.model_params = Some(params);
        self
    }
}

//T::Owned=T is a hack to make the bootstrapped samples fittable which only works for Array2
//Instead, for a general solution, we would like to write:
// - P: Fit<Array2<F>, T::Owned, Error>
// - type Object = EnsembleLearner<F, E, P::Object, T::Owned>;
//But this won't compile as it makes 'b unconstrained for some reason
impl<F: Clone, E, D, T, P: Fit<Array2<F>, T::Owned, Error>>
     Fit<ArrayBase<D, Ix2>, T, Error> for EnsembleLearnerParams<F, E, P>
where
    E: Copy + std::cmp::Eq + std::hash::Hash,
    D: Data<Elem = F> ,
    T: AsTargets<Elem = E> + FromTargetArrayOwned<E>,
    T::Owned: AsTargets,
{
    type Object = EnsembleLearner<F, E, P::Object, T::Owned>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, Error> {
        assert!(
            self.model_params.is_some(),
            "Must define an underlying model for ensemble learner",
        );

        let mut models = Vec::new();
        let rng =  &mut rand::thread_rng();

        let dataset_size = ((dataset.records.shape()[0] as f64) * self.bootstrap_proportion) as usize;

        //Had to modify lifetimes in impl_dataset to get this to work!
        let iter = dataset.bootstrap_samples(dataset_size, rng);

        let mut i = 0;
        for train in iter {
            println!("Fitting model {}, {}", i, dataset_size);
            i += 1;
            let model = self.model_params.as_ref().unwrap().fit(&train).unwrap();
            models.push(model);
            if models.len() == self.ensemble_size {
                break
            }
        }

        Ok(EnsembleLearner{models:models,
                           _t: PhantomData {}})

    }
}
