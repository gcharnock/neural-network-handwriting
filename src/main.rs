

struct Network {
    layers: Vec<Vec<f32>>
}

fn main() {
    println!("Hello, world!");
}

mod matrix {
    pub struct Matrix<T> {
        width: u32,
        values: Vec<T>
    }

    impl Matrix<T> {
        fn new(width: u32, height: u32) -> Matrix<T> {
            return Matrix {
                width,
                values: vec![0; width * height]
            }
        }

        fn set(&self, row: u32, column: u32, value: T) {
            self.values[column * self.with + row] = value;
        }

        fn get(&self, row: u32, column: u32) -> T {
            self.values[column * self.with + row]
        }
    }


    #[cfg(test)]
    mod tests {
        use matrix::Matrix;

        #[test]
        fn matrix_multiplication() {
            let mat = Matrix::new(3, 3);
        }
    }
}


