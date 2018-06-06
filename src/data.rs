
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
        return layer.iter().map(|v| (*v as f32 / 255.0)).collect();
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
