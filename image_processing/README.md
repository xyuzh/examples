# Process images

This example uses Ray Data to process the [ReLAION-2B](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) image dataset, which consists of over 2 billion rows. Each row consists of an image URL along with various metadata include a caption and image dimensions.

## Submit the job to Anyscale

Make sure you have your Huggingface credentials configured locally, then run:

```bash
./run.sh
```