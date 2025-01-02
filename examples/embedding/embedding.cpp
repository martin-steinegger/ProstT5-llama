#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <limits>
#include <climits>
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// static std::vector<std::string> split_lines(const std::string & s, const std::string & separator = "\n") {
//     std::vector<std::string> lines;
//     size_t start = 0;
//     size_t end = s.find(separator);

//     while (end != std::string::npos) {
//         lines.push_back(s.substr(start, end - start));
//         start = end + separator.length();
//         end = s.find(separator, start);
//     }

//     lines.push_back(s.substr(start)); // Add the last part

//     return lines;
// }

char number_to_char(unsigned int n) {
    switch(n) {
        case 0:  return 'A';
        case 1:  return 'C';
        case 2:  return 'D';
        case 3:  return 'E';
        case 4:  return 'F';
        case 5:  return 'G';
        case 6:  return 'H';
        case 7:  return 'I';
        case 8:  return 'K';
        case 9:  return 'L';
        case 10: return 'M';
        case 11: return 'N';
        case 12: return 'P';
        case 13: return 'Q';
        case 14: return 'R';
        case 15: return 'S';
        case 16: return 'T';
        case 17: return 'V';
        case 18: return 'W';
        case 19: return 'Y';
        default: return 'X'; // Default case for numbers not in the list
    }
}


// Function to read a FASTA file and store the sequences in a vector of pairs
std::vector<std::pair<std::string, std::string>> read_fasta(const std::string &filename) {
    std::vector<std::pair<std::string, std::string>> fastaSequences;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return fastaSequences;
    }

    std::string line;
    std::string header;
    std::string sequence;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '>') { // Header line
            if (!header.empty()) {
                // Store the previous sequence
                fastaSequences.emplace_back(header, sequence);
            }
            header = line.substr(1); // Remove '>'
            sequence.clear();
        } else {
            sequence += line; // Append sequence lines
        }
    }

    // Store the last sequence
    if (!header.empty()) {
        fastaSequences.emplace_back(header, sequence);
    }

    file.close();
    return fastaSequences;
}

static int encode(llama_context * ctx, std::vector<llama_token> & enc_input, std::string & result) {
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);
    // llama_set_embeddings(ctx, true);
    // run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        if (llama_encode(ctx, llama_batch_get_one(enc_input.data(), enc_input.size())) < 0) {
            LOG_ERR("%s : failed to encode\n", __func__);
        }
    } else { 
        LOG_ERR("%s : no encoder\n", __func__);
        return 1;
    }
    // Log the embeddings (assuming n_embd is the embedding size per token)
    // LOG_INF("%s: n_tokens = %zu, n_seq = %d\n", __func__, enc_input.size(), 1);
    float* embeddings = llama_get_embeddings(ctx);
    if (embeddings == nullptr) {
        LOG_ERR("%s : failed to retrieve embeddings\n", __func__);
        return 1;
    }
    int * arg_max_idx = new int[enc_input.size()];
    float * arg_max = new float[enc_input.size()];
    std::fill(arg_max, arg_max + enc_input.size(), std::numeric_limits<float>::lowest());
    int seq_len = enc_input.size() - 1;
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if(embeddings[i*seq_len + j] > arg_max[j]){
               arg_max_idx[j] = i;
               arg_max[j] = embeddings[i*seq_len + j];
            }
        }
    }
    for (int i = 0; i < seq_len; ++i) {
        result.push_back(number_to_char(arg_max_idx[i]));
    }
    // printf("\n");
    delete [] arg_max_idx;
    delete [] arg_max;
    return 0;
}


static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(-1);
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    }, NULL);
    common_params params;

    const char* model_file = argv[1];
    const char* fasta_file = argv[2];

    //if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
    // if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN)) {
    //     return 1;
    // }
    common_params_parser_init(params, LLAMA_EXAMPLE_EMBEDDING);
    // const common_params params_org = ctx_arg.params; // the example can modify the default params

    common_init();

    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.warmup = false;
    params.model = model_file;
    params.cpuparams.n_threads = 1;
    params.use_mmap = false;
    params.n_gpu_layers = 24;
    params.devices = parse_device_list(argv[3]);
    params.check_tensors = true;
    params.embedding = true;
    //params.devices = parse_device_list("none");
    // if (params.cpuparams.n_threads <= 0) {
    //     params.cpuparams.n_threads = std::thread::hardware_concurrency();
    // }
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // const int n_ctx_train = llama_n_ctx_train(model);
    // const int n_ctx = llama_n_ctx(ctx);

    // const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    //if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
    //    LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
    //    return 1;
    //}

    // if (n_ctx > n_ctx_train) {
    //     LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
    //             __func__, n_ctx_train, n_ctx);
    // }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // // split the prompt into lines
    // std::vector<std::string> prompts = split_lines(params.prompt, params.embd_sep);

    // // max batch size
    // const uint64_t n_batch = params.n_batch;
    // //GGML_ASSERT(params.n_batch >= params.n_ctx);

    // // tokenize the prompts and trim
    // std::vector<std::vector<int32_t>> inputs;
    // for (const auto & prompt : prompts) {
    //     auto inp = common_tokenize(ctx, prompt, true, true);
    //     if (inp.size() > n_batch) {
    //         LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
    //                 __func__, (long long int) inp.size(), (long long int) n_batch);
    //         return 1;
    //     }
    //     inputs.push_back(inp);
    // }
    // // check if the last token is SEP
    // // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    // for (auto & inp : inputs) {
    //     if (inp.empty() || inp.back() != llama_token_sep(model)) {
    //         LOG_WRN("%s: last token in the prompt is not SEP\n", __func__);
    //         LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
    //     }
    // }

    // // tokenization stats
    // if (params.verbose_prompt) {
    //     for (int i = 0; i < (int) inputs.size(); i++) {
    //         LOG_INF("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
    //         LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
    //         for (int j = 0; j < (int) inputs[i].size(); j++) {
    //             LOG("%6d -> '%s'\n", inputs[i][j], common_token_to_piece(ctx, inputs[i][j]).c_str());
    //         }
    //         LOG("\n\n");
    //     }
    // }

    // // initialize batch
    // const int n_prompts = prompts.size();

    // // count number of embeddings
    // int n_embd_count = 0;
    // if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
    //     for (int k = 0; k < n_prompts; k++) {
    //         n_embd_count += inputs[k].size();
    //     }
    // } else {
    //     n_embd_count = n_prompts;
    // }

    // allocate output
    // const int n_embd = llama_n_embd(model);
    // std::vector<float> embeddings(n_embd_count * n_embd, 0);
    std::string result;
    std::vector<std::pair<std::string, std::string>> fastaData = read_fasta(fasta_file);
    // std::string prompt;
    for (const auto &entry : fastaData) {
        const std::string &sequence = entry.second; 
        // prompt.clear();
        result.clear();
        if(sequence.size() > 1024)
            continue;
        printf("AA:  %s\n", sequence.c_str());

        std::vector<llama_token> embd_inp;
        embd_inp.reserve(sequence.length() + 2);
        embd_inp.emplace_back(llama_token_get_token(model, "<AA2fold>"));
        llama_token unk_aa = llama_token_get_token(model, "▁X");
        for (size_t i = 0; i < sequence.length(); ++i) {
            std::string current_char("▁");
            current_char.append(1, toupper(sequence[i]));
            llama_token token = llama_token_get_token(model, current_char.c_str());
            if (token == LLAMA_TOKEN_NULL) {
                embd_inp.emplace_back(unk_aa);
            } else {
                embd_inp.emplace_back(token);
            }
        }
        embd_inp.emplace_back(llama_token_get_token(model, "</s>"));

        // std::vector<llama_token> embd_inp = common_tokenize(ctx, prompt, true, true);
        encode(ctx, embd_inp, result);
        printf("3Di: %s\n", result.c_str());
    }


    //for (const auto & prompt : prompts) {
    //    embd_inp = common_tokenize(ctx, prompt, true, true);
    //    // encode if at capacity
    //    encode(ctx, embd_inp, result);
    //    printf("%s\n", result.c_str());
    //}

    llama_perf_context_print(ctx);

    // clean up
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
