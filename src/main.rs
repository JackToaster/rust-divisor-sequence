
use std::{collections::BTreeMap, time::Instant, cmp::min, fs::File, io::Write};
use galil_seiferas::gs_find;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use primefactor::PrimeFactors;


// Check if the last m numbers in a sequence repeat.
fn repeats_m(seq: &[usize], m: usize) -> Option<usize>
{
    if seq.len() < 1 {
        return None;
    }
    let subseq = &seq[seq.len() - m .. seq.len()];
    let remaining_seq = &seq[0 .. seq.len() - 1];

    // Search for the subsequence in const time with the Galil-Seiferas algorithm
    gs_find(remaining_seq, subseq)
}

// Number of factors cache to speed up computation.
// Since maximum values increase slowly, this could probably be a pre-computed array of the first ~10000 numbers to speed it up significantly.
type FacCache = BTreeMap<usize,usize>;

// Divisor function - returns the number of factors.
fn n_factors(num: usize, fac_cache: &mut FacCache) -> usize
{
    // Grab number of factors from the cache
    {
        if let Some(val) = fac_cache.get(&num) {
            // println!("{}: hit", num);
            return *val
        }
    }
    // Cache missed, calculate number of factors
    let prime_factors = PrimeFactors::from(num as u128);

    
    let num_factors = prime_factors.iter()
        .fold(
            1, 
            |n, factor| n * (1 + factor.exponent)
        );

    let num_factors: usize = num_factors.try_into().unwrap();

    // Save number of factors back to the cache
    {
        fac_cache.insert(num, num_factors);
    }

    // println!("{}: miss", num);
    return num_factors;
}



struct RResult<T> {
    sequence : Vec<T>,
    repeated : Option<Vec<T>>,
    repeated_at: Option<usize>,
}

// R(n,m) =
// the sum of the number of divisors of the last m elements
// Sequence starts with m 1's.
fn r(m: usize, max_length: usize,) -> RResult<usize>
{
    const CHECK_EVERY: usize = 100000;

    let mut fac_cache: FacCache = BTreeMap::<usize, usize>::new();
    // initialize sequence of m ones
    let mut seq = Vec::new();
    for _i in 0..m {
        seq.push(1);
    }

    // Keep adding terms until we find a repetition or hit the predefined length limit.
    while seq.len() < max_length {
        // Compute the next term
        let last_m = &seq[seq.len() - m .. seq.len()];
        // println!("m={}, last_m = {:?}", m, last_m);
        let last_m_factors = last_m.iter().map(|num| n_factors(*num, &mut fac_cache));
        let new_val = last_m_factors.sum();
        
        // Add value to the sequence
        seq.push(new_val);

        // Checking for repetitions is very expensive (especially once the sequence gets longer than a few million.)
        // So only check every CHECK_EVERY values.
        if seq.len() % CHECK_EVERY == 0 {
            // Look for a repetition
            if let Some(_n) = repeats_m(&seq, m) {
                // println!("Repetition detected, checking...");
                // Repetition is found. Now search for the start of it:
                let rep_pos = binary_repeat_search(&seq, m);
                let last_m = &seq[seq.len() - m .. seq.len()];
                // Return result
                return RResult {sequence: seq.clone(), repeated: Some(last_m.to_owned()), repeated_at: Some(rep_pos)};
            }
        }
    }
    
    // Check for repetitions one last time after we reach the length limit.
    if let Some(_n) = repeats_m(&seq, m) {
        // Find first repetition
        let rep_pos = binary_repeat_search(&seq, m);
        let last_m = &seq[seq.len() - m .. seq.len()];
        return RResult {sequence: seq.clone(), repeated: Some(last_m.to_owned()), repeated_at: Some(rep_pos)};
    }

    RResult { sequence: seq, repeated: None , repeated_at: None}
}

// Takes a sequence and finds the first repetition of m values
fn binary_repeat_search(seq: &Vec<usize>, m: usize) -> usize{
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
    let max_length = 100000000;
    // Range of values to test
    let trials = 1..=1000;

    // Process in batches this large, for the sake of parallelism.
    // Choose a number larger than your number of CPU cores for best performance.
    let batch_size = 16;

    // Open a file to write results
    let mut file = File::create("results_500.csv").unwrap();
    // write header to file
    let mut w = Vec::new();
    write!(w, "m, repeat_after, max_value\n").unwrap();
    file.write(&w).unwrap();

    // Record start time
    let program_start = Instant::now();

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
        let results: Vec<(usize, RResult<usize>)> = batch.into_par_iter()
            .map(
                |m| (m, r(m, max_length))
            ).collect();
        // print results and write to file
        for (m, result) in &results {
            let mut w = Vec::new();
            if let Some(_rep_seq) = &result.repeated {
                let max_val = result.sequence.iter().max().unwrap();
                println!("R(n,{}): Repeated @ n={}. Max val: {}", m, result.repeated_at.unwrap(), max_val);
                // println!("  Repeated sequence: {:?}", rep_seq);

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
    }

    let elapsed_time = program_start.elapsed();
    println!("Elapsed time: {} seconds", elapsed_time.as_secs_f64());
}
