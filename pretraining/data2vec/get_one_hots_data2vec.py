import os
import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import sys
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def onehot_embedder(seq):
    """
    Convert a sequence into a one-hot encoded matrix.
    If the sequence length is not a multiple of 1000, pad with zeros at the end.
    """
    # Backward-compatible default used in the original script.
    return onehot_embedder_with_args(seq, segment_len=4096)


def onehot_embedder_with_args(seq: str, segment_len: int) -> np.ndarray:
    onehot = np.zeros((len(seq), 4), dtype=np.int8)
    seq_array = np.array(list(seq))
    onehot[seq_array == "A", 0] = 1
    onehot[seq_array == "C", 1] = 1
    onehot[seq_array == "G", 2] = 1
    onehot[seq_array == "T", 3] = 1

    if len(seq) % segment_len != 0:
        pad = segment_len - len(seq) % segment_len
        onehot = np.concatenate([onehot, np.zeros((pad, 4), dtype=np.int8)], axis=0)
    return onehot


def process_genome(gid: str, fna_dir: str, segment_len: int):
    """
    Reads a FASTA file for a given genome ID and returns a list of one-hot encoded matrices,
    one for each contig in the FASTA.
    """
    fna_file = os.path.join(fna_dir, f"{gid}.fna")
    embeddings = []
    for record in SeqIO.parse(fna_file, "fasta"):
        seq = str(record.seq).upper()
        embedding = onehot_embedder_with_args(seq, segment_len)
        embeddings.append(embedding)
    return embeddings


def process_genome_wrapper(gid: str, fna_dir: str, segment_len: int, max_segments: int, out_onehot_dir: str, out_split_indices_dir: str):
    """
    Processes a genome by first converting each contig into one-hot encoded format,
    then applying sliding window segmentation (window size=2000, stride=1000) for each contig.
    Finally, appends a special 1000x4 segment to the end of each contig to serve as a marker.
    The resulting segments are saved to a .npy file.
    """
    #if the file already exists, skip
    # if os.path.exists(f"./one_hots_data2vec_new/{gid}.npy") and os.path.exists(f"./one_hots_split_indices_data2vec_new/{gid}.npy"):
    #     return (gid, True)
    
    try:
        embs = process_genome(gid, fna_dir=fna_dir, segment_len=segment_len)
        emb_chunks = []
        end_indices = []
        lengths = 0
        for emb in embs:
            emb = np.squeeze(emb)
            added = False
            for i in range(0, emb.shape[0] - segment_len + 1, segment_len):
                if lengths <= max_segments - 1:
                    added = True
                    emb_chunks.append(emb[i : i + segment_len])
                    lengths += 1
            # Append special separator segment for this contig
            end_indices.append(lengths - 1)
    
        emb_chunks = np.array(emb_chunks)
        os.makedirs(out_onehot_dir, exist_ok=True)
        os.makedirs(out_split_indices_dir, exist_ok=True)

        indices_file = os.path.join(out_split_indices_dir, f"{gid}.npy")
        np.save(indices_file, np.array(end_indices))

        out_file = os.path.join(out_onehot_dir, f"{gid}.npy")
        np.save(out_file, emb_chunks)
        return (gid, True)
    except Exception as e:
        sys.stderr.write(f"Error processing {gid}: {str(e)}\n")
        return (gid, False)


def restart_script(signum, frame):
    print(f"Signal {signum} received. Restarting the script...")
    os.execv(sys.executable, ['python'] + sys.argv)

def main():
    parser = argparse.ArgumentParser(description="Create one-hot genome segments for GenomeData2Vec.")
    parser.add_argument("--genome_ids_file", type=str, required=True, help="CSV/XLSX containing a 'genome_id' column.")
    parser.add_argument("--sheet_name", type=str, default="genome_metadata", help="Excel sheet name (if --genome_ids_file is .xlsx).")
    parser.add_argument("--start_index", type=int, default=0, help="Start index into the genome_id list (useful for sharding).")
    parser.add_argument("--fna_dir", type=str, required=True, help="Directory containing per-genome FASTA files named <genome_id>.fna")
    parser.add_argument("--out_onehot_dir", type=str, default="./one_hots_data2vec_new", help="Output directory for one-hot segments.")
    parser.add_argument("--out_split_indices_dir", type=str, default="./one_hots_split_indices_data2vec_new", help="Output directory for contig split indices.")
    parser.add_argument("--segment_len", type=int, default=4096, help="Segment length (also used for padding).")
    parser.add_argument("--max_segments", type=int, default=8192, help="Maximum number of segments per genome.")
    parser.add_argument("--num_workers", type=int, default=10, help="Parallel workers.")
    args = parser.parse_args()

    if args.genome_ids_file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.genome_ids_file, sheet_name=args.sheet_name, usecols="A", dtype=str)
    else:
        df = pd.read_csv(args.genome_ids_file, dtype=str)
    if "genome_id" not in df.columns:
        raise ValueError("Expected a 'genome_id' column in --genome_ids_file")

    genome_ids = df["genome_id"].astype(str).tolist()
    genome_ids = genome_ids[args.start_index :]

    print(f"Using {args.num_workers} parallel workers on {len(genome_ids)} genomes.")

    # Register signal handlers
    signal.signal(signal.SIGTERM, restart_script)
    signal.signal(signal.SIGINT, restart_script)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_genome_wrapper,
                gid,
                args.fna_dir,
                args.segment_len,
                args.max_segments,
                args.out_onehot_dir,
                args.out_split_indices_dir,
            ): gid
            for gid in genome_ids
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            gid, success = future.result()
            if not success:
                print(f"Failed to process {gid}")


if __name__ == "__main__":
    main()
