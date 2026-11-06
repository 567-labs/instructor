# What to Expect
This script demonstrates how to use the `Instructor` library for fine-tuning a Python function that performs three-digit multiplication. It uses Pydantic for type validation and logging features to generate a fine-tuning dataset.

## How to Run

### Prerequisites
- Python 3.9
- `Instructor` library

### Steps
1. **Install Dependencies**  
   If you haven't already installed the required libraries, you can do so using pip:
    ```
    pip install instructor pydantic
    ```

2. **Set Up Logging**  
   The script uses Python's built-in `logging` module to log the fine-tuning process. Ensure you have write permissions in the directory where the log file `math_finetunes.jsonl` will be saved.

3. **Run the Script**  
    Navigate to the directory containing `script.py` and run it:
    ```
    python three_digit_mul.py
    ```

    This will execute the script, running the function ten times with random three-digit numbers for multiplication. The function outputs and logs are saved in `math_finetunes.jsonl`.

4. **Fine-Tuning**  
    Once you have the log file, you can run a fine-tuning job using the following `Instructor` CLI command:
    ```
    instructor jobs create-from-file math_finetunes.jsonl
    ```
    Wait for the fine-tuning job to complete.

    If you have validation date you can run:

    ```
    instructor jobs create-from-file math_finetunes.jsonl --n-epochs 4 --validation-file math_finetunes_val.jsonl 
    ```

### Output

That's it! You've successfully run the script and can now proceed to fine-tune your model.

### Dispatch 

Once you have the model you can replace the model in `three_digit_mul_dispatch.py` with the model you just fine-tuned and run the script again. This time, the script will use the fine-tuned model to predict the output of the function.