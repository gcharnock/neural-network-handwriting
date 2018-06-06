#[cfg_attr(test, macro_use)]
extern crate assert_approx_eq;

extern crate byteorder;
extern crate rand;
extern crate num_traits;

mod network;
mod linear;
mod data ;
use network::{Network, NetworkLayer};
use linear::print_vector;
use rand::Rand;


fn make_expected(label: u8) -> Vec<f32> {
    let mut expected = Vec::with_capacity(10);
    for i in 0..10 {
        if label == i {
            expected.push(1.0);
        } else {
            expected.push(0.0);
        }
    }
    return expected;
}


fn main() {
    let mut gen_w = rand::thread_rng();
    let mut gen_b = rand::thread_rng();

    let mut gen_weights = |_, _| {
        let r: f32 = Rand::rand(&mut gen_w);
        r - 0.5f32
    };

    let mut gen_biases = |_| {
        let r: f32 = Rand::rand(&mut gen_b);
        r - 0.5f32
    };

    let training_data = data::read_training_data();
    let input_to_hidden = NetworkLayer::<f32>::new(
        training_data.len(), 30,
        &mut gen_weights, &mut gen_biases);

    let hidden_to_output = NetworkLayer::<f32>::new(
        30, 10,
        &mut gen_weights, &mut gen_biases);

    let layers = vec![input_to_hidden, hidden_to_output];
    let mut network = Network::new(layers);

    network.describe();

    for i in 0..1 {
        let this_data = training_data.unpack_layer_to_f32(i);
        let expected = make_expected(training_data.labels[i]);

        network.set_input_activations(&this_data);
        network.propagate();

        for activations in network.activations.iter() {
            print!("activations = ");
            print_vector(&activations);
            println!();
        }

        network.back_propagate(&expected);

        for errors in network.activation_errors.iter() {
            print!("activation_error = ");
            print_vector(&errors);
            println!();
        }
    }
}

