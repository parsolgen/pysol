import transform(in int count, in int src_n1, in Array<float> src[src_n1], in int dst_n1, out Array<double> dst[dst_n1]);
import normalize(in int src_n1, in Array<double> src[src_n1], in int result_n1, out Array<double> result[result_n1]);

export sub process_data()
{
    Array<float[32]> input_arr[64];
    Array<double[32]> output_arr[64];
    for (int i = 0; i < 64; i++) {
        transform(i, input_arr{0}[i], output_arr{0}[i]);
        if (i < 32) {
            normalize(output_arr{0}[i], output_arr{0}[i]);
        }
    }
}
