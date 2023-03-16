
use std::{time::Instant, cmp::min, fs::File, io::Write, sync::Arc};
use galil_seiferas::gs_find;
use rayon::{prelude::{IntoParallelIterator, ParallelIterator}, vec};
use primefactor::PrimeFactors;


// Check if the last m numbers in a sequence repeat.
fn repeats_m<T>(seq: &[T], m: usize) -> Option<usize>
    where T: Eq
{
    if seq.len() < 1 {
        return None;
    }
    let subseq = &seq[seq.len() - m .. seq.len()];
    let remaining_seq = &seq[0 .. seq.len() - 1];

    // Search for the subsequence in const time with the Galil-Seiferas algorithm
    gs_find(remaining_seq, subseq)
}

// Divisor function - returns the number of factors.
fn build_fac_table(length: usize) -> Vec<u8>
{
    (0..length).map(|num|
        {
            let prime_factors = PrimeFactors::from(num as u128);

    
            let num_factors = prime_factors.iter()
                .fold(
                    1, 
                    |n, factor| n * (1 + factor.exponent)
                );

            num_factors.try_into().unwrap()
        }
    ).collect()
}



struct RResult<T> {
    sequence : Vec<T>,
    repeated : Option<Vec<T>>,
    repeated_at: Option<usize>,
}

// R(n,m) =
// the sum of the number of divisors of the last m elements
// Sequence starts with m 1's.
fn r(m: usize, max_length: usize, fac_table: Arc<Vec<u8>>) -> RResult<u16>
{
    if m == 0 {
        return RResult {sequence: vec![], repeated: None, repeated_at: None};
    }

    const CHECK_EVERY: usize = 1000000;

    // initialize sequence of m ones
    let mut seq: Vec<u16> = Vec::new();
    let mut div_num_seq: Vec<usize> = Vec::new();
    let mut div_sum: usize = m.try_into().unwrap();
    for _i in 0..m {
        seq.push(1);
        div_num_seq.push(1);
    }

    // Keep adding terms until we find a repetition or hit the predefined length limit.
    while seq.len() < max_length {
        for _i in 0..CHECK_EVERY {
            // println!("{}", div_sum);
            // Add value to the sequence
            seq.push(div_sum as u16);
            // Get number of divisors for new value
            let new_n_divisors: usize = fac_table[div_sum as usize].into();

            // Subtract old value from sum
            div_sum -= div_num_seq[seq.len() % m];
            // Update sum by adding newly added value and subtracting old value
            div_sum += new_n_divisors;

            // Add to div_num_seq
            div_num_seq[seq.len() % m] = new_n_divisors as usize;

        }
        

        // Checking for repetitions is very expensive (especially once the sequence gets longer than a few million.)
        // So only check every CHECK_EVERY values.

        // Look for a repetition
        if let Some(_n) = repeats_m(&seq, m) {
            // println!("Repetition detected, checking...");
            // Repetition is found. Now search for the start of it:
            let rep_pos = binary_repeat_search(&seq, m) - m;
            let last_m = &seq[seq.len() - m .. seq.len()];
            // Return result
            return RResult {sequence: seq.clone(), repeated: Some(last_m.to_owned()), repeated_at: Some(rep_pos)};
        }
    }

    RResult { sequence: seq, repeated: None , repeated_at: None}
}

// Takes a sequence and finds the first repetition of m values
fn binary_repeat_search<T>(seq: &Vec<T>, m: usize) -> usize
    where T: Eq
{
    let mut low: usize = 0;
    let mut high: usize = seq.len();

    while low < high {
        let mid = ((high - low) / 2) + low;
        // println!("{} {} {}", low, mid, high);
        
        // If a repetition is found, search earlier in the sequence. If none found, search later.
        if let Some(_n) = repeats_m(&seq[0 .. mid], m) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;

}

fn main() {
    // Set the maximum sequence length before giving up.
    // WARNING: 100,000,000 will take a long time (half an hour or more, depending on your computer)
    let max_length = 1_000_000_000;
    // Range of values to test
    let trials = 1..=2000;

    // Process in batches this large, for the sake of parallelism.
    // Choose a number larger than your number of CPU cores for best performance.
    // However, for large values of m, you may need to decrease it depending on available memory.
    let batch_size = 8;

    // Open a file to write results
    let mut file = File::create("results_500.csv").unwrap();
    // write header to file
    let mut w = Vec::new();
    write!(w, "m, repeat_after, max_value\n").unwrap();
    file.write(&w).unwrap();

    // Record start time
    let program_start = Instant::now();


    let fac_table = Arc::new(build_fac_table(32768));

    // Run trials in batches
    let mut start_val = *trials.start();
    while start_val <= *trials.end() {
        // Get values for this batch
        let batch = start_val .. min(start_val + batch_size, *trials.end() + 1);
        start_val += batch_size;
        println!("\nStarting batch: {:?}", batch);

        // Record batch start time
        let batch_start = Instant::now();
        // Calculate results in parallel
        let results: Vec<(usize, RResult<u16>)> = batch.into_par_iter()
            .map(
                |m| (m, r(m, max_length, fac_table.clone()))
            ).collect();
        // print results and write to file
        for (m, result) in &results {
            let mut w = Vec::new();
            if let Some(_rep_seq) = &result.repeated {
                let max_val = result.sequence.iter().max().unwrap();
                println!("R(n,{}): Repeated @ n={}. Max val: {}.", m, result.repeated_at.unwrap(), max_val);
                // Warning - sequences can be very long! printing them might be a bad idea.
                // println!("  sequence: {:?}", result.sequence);

                write!(w, "{}, {}, {}\n", m, result.repeated_at.unwrap(), max_val).unwrap();
            } else {
                println!("R(n,{})", m);
                println!("!!!!  No repeat for n<{}", max_length);
                write!(w, "{}, None, None\n", m).unwrap();
            }
            file.write(&w).unwrap();
        }
        let elapsed_time = batch_start.elapsed();
        println!("Batch finished in {} seconds", elapsed_time.as_secs_f64());
        file.flush();
    }

    let elapsed_time = program_start.elapsed();
    println!("Elapsed time: {} seconds", elapsed_time.as_secs_f64());
}
