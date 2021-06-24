from argparse import ArgumentParser
import torch, flair

# dataset, model and embedding imports
from flair.datasets import UniversalDependenciesCorpus, XTREME
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

if __name__ == "__main__":

    # All arguments that can be passed
    parser = ArgumentParser()
    parser.add_argument("-s", "--seeds", nargs='+', type=int, default='42')  # pass list of seeds for experiments
    parser.add_argument("-c", "--cuda", type=int, default=0, help="CUDA device")  # which cuda device to use
    parser.add_argument("-m", "--model", type=str, help="Model name (such as Hugging Face model hub name")
    parser.add_argument("-d", "--dataset", type=str, help="Defines dataset, choose between imst, boun or xtreme")

    # Parse experimental arguments
    args = parser.parse_args()

    # use cuda device as passed
    flair.device = f'cuda:{str(args.cuda)}'

    # for each passed seed, do one experimental run
    for seed in args.seeds:
        flair.set_seed(seed)

        # model
        hf_model = args.model

        # initialize embeddings
        embeddings = TransformerWordEmbeddings(
            model=hf_model,
            layers="-1",
            subtoken_pooling="first",
            fine_tune=True,
            use_context=False,
            respect_document_boundaries=False,
        )

        # select dataset depending on which language variable is passed

        tag_type = None

        if args.dataset in ["imst", "boun"]:
            tag_type = "upos"
            corpus = UniversalDependenciesCorpus(data_folder="./data",
                                                train_file=f"tr_{args.dataset}-ud-train.conllu",
                                                dev_file=f"tr_{args.dataset}-ud-dev.conllu",
                                                test_file=f"tr_{args.dataset}-ud-test.conllu")
        elif args.dataset == "xtreme":
            tag_type = "ner"
            corpus = XTREME(languages="tr")

        # make the dictionary of tags to predict
        tag_dictionary = corpus.make_tag_dictionary(tag_type)

        # init bare-bones sequence tagger (no reprojection, LSTM or CRF)
        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )

        # init the model trainer
        trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

        # make string for output folder
        output_folder = f"flert-{args.dataset}-{hf_model}-{seed}"

        # train with XLM parameters (AdamW, 20 epochs, small LR)
        from torch.optim.lr_scheduler import OneCycleLR

        trainer.train(
            output_folder,
            learning_rate=5.0e-5,
            mini_batch_size=16,
            mini_batch_chunk_size=1,
            max_epochs=10,
            scheduler=OneCycleLR,
            embeddings_storage_mode='none',
            weight_decay=0.,
            train_with_dev=False,
            use_final_model_for_eval=True
        )


