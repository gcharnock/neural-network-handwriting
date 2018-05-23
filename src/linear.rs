use num_traits::Float;
use std::ops::Mul;
use std::ops::Add;
use std::ops::AddAssign;
use std::fmt::Display;

pub fn vec_add<T: Add + Float + Clone>(a: &Vec<T>, b: &Vec<T>) -> Vec<T>
    where
        T: Add<Output=T> {
    if a.len() != b.len() {
        panic!("cannot add vectors of different lengths");
    }

    let mut out = vec![T::zero(); a.len()];
    for i in 0..a.len() {
        out[i] = a[i].clone() + b[i].clone();
    }
    return out;
}

pub struct Matrix<T> {
    cols: usize,
    rows: usize,
    values: Vec<T>,
}

impl<T: Clone + Mul + AddAssign + Float + Display> Matrix<T>
    where
        T: Mul<Output=T> {
    pub fn new<F>(rows: usize, cols: usize, init: &mut F) -> Matrix<T>
        where F: FnMut() -> T {
        let len = rows * cols;
        let mut values = Vec::with_capacity(len);
        for _i in 0..len {
            let v = init();
            values.push(v);
        }
        Matrix { rows, cols, values }
    }

    fn set(&mut self, row: usize, col: usize, value: T) {
        self.values[col * self.rows + row] = value;
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.values[col * self.rows + row].clone()
    }

    fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{} ", self.get(row, col));
            }
            println!();
        }
    }

    pub fn multiply_vec(&self, vec: &Vec<T>) -> Vec<T> {
        if self.cols != vec.len() {
            panic!("invalid dimentions");
        }
        let mut out = vec![T::zero(); self.rows];
        for col in 0..self.cols {
            for row in 0..self.rows {
                out[row] += self.get(row, col) * vec[col].clone();
            }
        }
        out
    }
}


#[cfg(test)]
mod tests {
    use linear::Matrix;

    #[test]
    fn matrix_vector_multiplication() {
        let mut mat = Matrix::<i32>::new(2, 3, || 0);
        mat.set(0, 0, 1);
        mat.set(0, 1, 2);
        mat.set(0, 2, 3);
        mat.set(1, 0, 4);
        mat.set(1, 1, 5);
        mat.set(1, 2, 6);

        let vec: Vec<i32> = vec![1, 2, 3];

        let v_out = mat.multiply_vec(&vec);

        assert_eq!(v_out.len(), 2);

        assert_eq!(v_out[0], 14);
        assert_eq!(v_out[1], 32);
    }
}
