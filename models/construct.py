from models.GraphCL import GraphCL
from models.BGRL_G2L import BGRL_G2L
from models.GRACE import Grace
from models.DGI_transductive import DGI
from models.BGRL import BGRL

def model_construct_local(args, model_name, data, device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    elif(args.dataset == 'ogbn-arxiv'):
        use_ln = True
        layer_norm_first = True
    else:
        use_ln = False
        layer_norm_first = False

    if(model_name == 'GRACE'):
        model = Grace(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.lr, \
                      weight_decay=args.weight_decay, \
                      tau=args.tau, \
                      layer=args.encoder_layer,\
                      device=device,\
                      use_ln=use_ln,\
                      layer_norm_first=layer_norm_first)
    elif(model_name == 'DGI'):
        model = DGI(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,tau=args.tau, layer=args.encoder_layer,device=device,
                use_ln=use_ln,
                layer_norm_first=layer_norm_first)
    elif(model_name == 'BGRL'):
        model = BGRL(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,tau=args.tau, layer=args.encoder_layer,device=device)
    return model.to(device)

 
def model_construct_global(args,model_name,dataset,device):
    if(model_name == 'BGRL-G2L'):
        model = BGRL_G2L(args = args, \
                      nfeat = dataset.num_features, \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      dropout=args.dropout, \
                      lr=args.lr, \
                      weight_decay=args.weight_decay, \
                      tau=args.tau, \
                      layer=args.encoder_layer,\
                      device=device)
    elif(model_name == 'GraphCL'):
        model = GraphCL(args = args, \
                      nfeat = dataset.num_features, \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      dropout=args.dropout, \
                      lr=args.lr, \
                      weight_decay=args.weight_decay, \
                      tau=args.tau, \
                      layer=args.encoder_layer,\
                      device=device)
    return model.to(device)