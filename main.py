import argparse
import constant as C
from Utils.log import Log
from Utils.Timer import Timer
import Algorithm.FoCore as FoCore
import Algorithm.FoCache as FoCoreExt
from MLGraph.MLGraph import MLGraph

if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='FocusCore Decomposition of Multilayer Graphs')

    # arguments
    parser.add_argument('-a', type=str, dest="action", default="Decomposition", help='Action, Decomposition')
    parser.add_argument('-m', type=str, dest="algorithm", default="diff", help='Algorithm, FirmCore, FirmTruss')
    parser.add_argument('-d', type=str, dest="dataset", default="homo", help='Dataset name')
    # options
    parser.add_argument('--save', dest='save', action='store_true', default=True, help='save results')

    # read the arguments
    args = parser.parse_args()

    # dataset path
    C.init_dir()
    dataset_path = C.DATASET_DIR / f"{args.dataset}.txt"
    L = Log(__name__).get_logger()
    L.info("\n" + "#" * 32 + f"\n\t\t#{args.action}\n\t\t#Algo:{args.algorithm}\n\t\t#Dataset:{args.dataset}"
           + "\n" + "#" * 32)

    # create the input graph and print its name
    timer = Timer()
    timer.start()
    L.info("Loading Graph...")
    multilayer_graph = MLGraph(dataset_path)
    timer.stop()
    L.info("Loading Phase: ")
    timer.print_timer()
    multilayer_graph.order_layer_by_density()
    if args.action == 'Decomposition':
        if args.algorithm == "EP":
            L.info('---------- FoCore Decomposition-EP ----------')
            FoCore.focore_decomposition_enum_pruning(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                                                     multilayer_graph.ordered_layers_iterator, args.dataset,
                                                     save=args.save)
        elif args.algorithm == "IP":
            L.info('---------- FoCore Decomposition-IP ----------')
            FoCore.focore_decomposition_interleaved_peeling(multilayer_graph.adjacency_list,
                                                            multilayer_graph.nodes_iterator,
                                                            multilayer_graph.ordered_layers_iterator, args.dataset,
                                                            save=args.save)
        elif args.algorithm == "VC":
            L.info('---------- FoCore Decomposition-VC ----------')
            FoCore.focore_decomposition_vc(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                                           multilayer_graph.ordered_layers_iterator, args.dataset,
                                           save=args.save)

        elif args.algorithm == "diff":
            L.info('---------- Check correctness of FoCore Decomposition ----------')
            FoCore.check_diff(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                              multilayer_graph.ordered_layers_iterator, args.dataset, save=args.save)

        else:
            L.info('---------- FoCore Decomposition ----------')
            FoCore.focore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                                        multilayer_graph.layers_iterator, args.dataset, save=args.save)

    elif args.action == 'Cache':
        if args.algorithm == "diff":
            L.info('---------- Check correctness of FoCore Index ----------')
            FoCoreExt.check(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator,
                            multilayer_graph.ordered_layers_iterator, args.dataset)

    elif args.action == 'Subgraph':
        L.info('---------- FoCore Densest Subgraph Search ----------')
        FoCore.focore_denest_graph(multilayer_graph, args.dataset)

    elif args.action == 'Analyze':
        L.info('---------- Analyze Core ----------')
        if args.algorithm == "all":
            L.info('---------- FoCore ----------')
            FoCore.analyze_core(multilayer_graph, args.dataset)
        if args.algorithm == "den":
            FoCore.core_den(multilayer_graph, args.dataset)

    else:
        parser.print_help()
