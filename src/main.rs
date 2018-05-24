extern crate byteorder;
extern crate rand;
extern crate num_traits;

mod real {
    use std::f32;
    use std::ops::Neg;
    use std::ops::Add;
    use std::ops::Div;
    use std::ops::Mul;
    use num_traits::Float;

    pub trait Real
        where Self: Sized + Neg<Output=Self> +
        Add<Output=Self> + Mul<Output=Self> + Div<Output=Self> {
        fn exp(x: Self) -> Self;
    }

    impl Real for f32 {
        fn exp(x: f32) -> f32 {
            return f32::exp(x);
        }
    }

    pub fn sigmoid<T: Real + Float + Neg + Add + Div>(x: T) -> T
        where T: Neg<Output=T> + Add<Output=T> + Div<Output=T> {
        T::one() / (T::one() + Real::exp(-x))
    }

    pub fn sigmoid_prime<T: Neg + Add + Div + Float>(x: T) -> T
        where T: Neg<Output=T> + Add<Output=T> + Div<Output=T>  {
        let exp_neg_x = T::exp(-x);
        let one_minus_exp_neg_x = T::one() - exp_neg_x;
        return -exp_neg_x / one_minus_exp_neg_x / one_minus_exp_neg_x;
    }
}

mod network;
mod linear;

mod data {
    use std::fs::File;
    use std::io::Read;
    use byteorder::{BigEndian, ReadBytesExt};

    pub struct TrainingData {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<Vec<u8>>,
        pub labels: Vec<u8>,
    }

    impl TrainingData {
        pub fn len(&self) -> usize {
            return self.rows * self.cols;
        }

        pub fn unpack_layer_to_f32(&self, layer_number: usize) -> Vec<f32> {
            let layer = &self.data[layer_number];
            return layer.iter().map(|v| *v as f32).collect();
        }
    }

    pub fn read_training_data() -> TrainingData {
        let mut label_file = File::open("data/train-labels-idx1-ubyte").unwrap();
        let label_magic_number = label_file.read_i32::<BigEndian>().unwrap();
        let label_item_count = label_file.read_i32::<BigEndian>().unwrap() as usize;

        let mut label_out = Vec::<u8>::with_capacity(label_item_count);

        if label_magic_number != 2049 {
            panic!("did got get expected magic number from file. Got {}", label_magic_number);
        }

        let mut file = File::open("data/train-images-idx3-ubyte").unwrap();
        let magic_number = file.read_i32::<BigEndian>().unwrap();
        let item_count = file.read_i32::<BigEndian>().unwrap() as usize;
        let rows = file.read_i32::<BigEndian>().unwrap() as usize;
        let cols = file.read_i32::<BigEndian>().unwrap() as usize;
        let pixels = rows * cols;
        let mut out = Vec::<Vec<u8>>::with_capacity(item_count);

        if magic_number != 2051 {
            panic!("did got get expected magic number from file");
        }

        if label_item_count != item_count {
            panic!("item counts did not match between data and label file");
        }

        println!("item count = {}, rows = {}, cols = {}", item_count, rows, cols);
        for i in 0..item_count {
            let mut data = Vec::<u8>::with_capacity(pixels);
            unsafe { data.set_len(pixels); }
            let bytes_read = file.read(&mut data[..]).unwrap();
            if bytes_read != pixels {
                panic!("Unable to read full image (read {}, expected {})", bytes_read, pixels);
            }

            let label = label_file.read_u8().unwrap();
            if i < 3 {
                println!("should be {}", label);
                print_image(cols, &data);
            }
            out.push(data);
            label_out.push(label);
        }
        return TrainingData {
            rows,
            cols,
            data: out,
            labels: label_out,
        };
    }

    pub fn print_image(cols: usize, image: &Vec<u8>) {
        let rows = image.len() / cols;
        for row in 0..rows {
            for col in 0..cols {
                let v = image[cols * row + col];
                if v > 240 {
                    print!("#");
                } else if v > 220 {
                    print!("*");
                } else if v > 128 {
                    print!(".");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
    }
}

use std::fmt::Display;

fn sum_of_squared_error(label: u8, output: &Vec<f32>) -> f32 {
    let label = label as usize;
    let mut total = 0.0;
    for i in 0..output.len() {
        if i == label {
            total += (output[i] - 1.0) * (output[i] - 1.0);
        } else {
            total += output[i] * output[i];
        }
    }
    return total;
}

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

fn print_vector<T: Display>(vec: &Vec<T>) {
    for c in vec.iter() {
        print!("{} ", c);
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let training_data = data::read_training_data();
    let input_to_hidden = network::NetworkLayer::<f32>::new(training_data.len(), 30, &mut rng);
    let hidden_to_output = network::NetworkLayer::<f32>::new(30, 10, &mut rng);

    for i in 0..1 {
        let this_data = training_data.unpack_layer_to_f32(i);

        let mut hidden_activation = vec![0.0; input_to_hidden.num_outputs];
        input_to_hidden.eval_layer(&this_data, &mut hidden_activation);

        let mut output_activation = vec![0.0; input_to_hidden.num_outputs];
        hidden_to_output.eval_layer(&hidden_activation, &mut output_activation);

        print!("output:");
        print_vector(&output_activation);
        println!(" error = {}", sum_of_squared_error(training_data.labels[i], &output_activation));

        let expected = make_expected(training_data.labels[i]);
        let sigma_last = hidden_to_output.find_output_sigma(&output_activation, &expected);

        print!("sigma last: ");
        print_vector(&sigma_last);
        println!();

        let sigma_hidden =
            hidden_to_output.back_propagate_layer(&hidden_activation, &sigma_last, &expected);

        print!("sigma hidden: ");
        print_vector(&sigma_hidden);
        println!();

        let sigma_first =
            input_to_hidden.back_propagate_layer(&this_data, &sigma_hidden, &expected);

        print!("sigma first: ");
        print_vector(&sigma_first);
        println!();

        println!();
    }
}

