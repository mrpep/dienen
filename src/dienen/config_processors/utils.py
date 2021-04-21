import copy

symbols = {'global': '$',
           'no-cachable': '!'}

def single_elem_list_to_elem(list_elem):
    if isinstance(list_elem,list) and len(list_elem) == 1:
        elem = list_elem[0]
        elem = single_elem_list_to_elem(elem)
    elif isinstance(list_elem,list):
        elem = [single_elem_list_to_elem(elem_i) for elem_i in list_elem]
    else:
        elem = list_elem
    return elem

def find_name(base_name,used_names):
    name_not_found = 1
    i = 0
    while name_not_found:
        layer_name = '{}_{}'.format(base_name,i)
        if layer_name not in used_names:
            name_not_found = 0
        i+=1
    return layer_name

def name_layer(layer_type,layer_config,names,base_name=None,logger=None):
    #Give name to layer:
    layer_name = layer_config.get('name',None)
    if base_name:
        layer_type = '{}/{}'.format(base_name,layer_type)
    if layer_name in names:
        old_name = layer_name
        layer_name = find_name(layer_type,names)
        if logger:
            logger.debug('Name {} already exists. Layer will be renamed to {}'.format(old_name,layer_name))
    elif layer_name is None:
        layer_name = find_name(layer_type,names)
        if logger:
            logger.debug('Automatically assigned name {} to unnamed layer'.format(layer_name))
    layer_config['name'] = layer_name
    names.append(layer_name)

    return layer_config, names

def assign_names_and_inputs(layerlist,names=None,base_name=None,logger=None):
    if not names:
        names = []
    #Keep track of last input
    last_layer = 'x'
    alllayer_inputs = {}
    
    for layer in layerlist:
        layer_type = list(layer.keys())[0]
        layer_config = layer[layer_type]
        layer_config, names = name_layer(layer_type,layer_config,names,base_name=base_name,logger=logger)
        layer_config['class'] = layer_type

        #By default, if input is unspecified, last layer will be considered input.
        if 'input' not in layer_config and last_layer:
            layer_config['input'] = last_layer
            if logger:
                logger.debug('Automatically set {} as input of layer {}'.format(last_layer,layer_config['name']))
        elif 'input' not in layer_config and not last_layer:
            raise Exception('At least the first layer should have an input')
        elif 'input' in layer_config:
            layer_ins = layer_config['input']
            layer_ins = [layer_ins] if not isinstance(layer_ins,list) else layer_ins
            layer_ins = [single_elem_list_to_elem(alllayer_inputs[k.split('->')[0]]) if '->input' in k else k for k in layer_ins]
            layer_config['input'] = layer_ins

        alllayer_inputs[layer_config['name']] = layer_config['input']
        last_layer = layer_config['name']

    return layerlist, names

def layerlist_to_layerdict(layerlist,names=None,base_name=None,logger=None):

    layerlist, names = assign_names_and_inputs(layerlist,names=None,base_name=base_name,logger=logger)
    new_config = {}
    for layer in layerlist:
        layer_type = list(layer.keys())[0]
        layer_config = layer[layer_type]

        layer_name = layer_config.pop('name')
        new_config[layer_name] = layer_config
        last_layer = layer_name

    return new_config, names

def replace_inputs_by_hierarchy(unfolded,hierarchy):
    unfolded = copy.deepcopy(unfolded)
    
    def recursive_replace(names,hierarchy):
        return [recursive_replace(hierarchy[k]['outputs'], hierarchy) if k in hierarchy else k for k in names]

    for layer_name, layer_params in unfolded.items():
        layer_in = layer_params['input']
        if not isinstance(layer_in,list):
            layer_in = [layer_in]
        new_input = [recursive_replace(hierarchy[k]['outputs'], hierarchy) if k in hierarchy else k for k in layer_in]
        if len(new_input)==1:
            new_input = new_input[0]

        layer_params['input'] = new_input
    return unfolded

def gather_blocks(config):
    blocks = {} 
    for layer_name, layer_config in config.items(): 
        if layer_config['class'] == 'BlockDefinition': 
            blocks[layer_name] = layer_config

    for block_name, block_config in blocks.items():
        config.pop(block_name) 

    return blocks

def uniformize_inputs(unfolded):
    unfolded_ = copy.deepcopy(unfolded)
    for layer_name, layer_config in unfolded_.items():
        layer_config['input'] = single_elem_list_to_elem(layer_config['input'])
    return unfolded_

def recursive_global_replace(tree,global_config):
    if isinstance(tree,dict):
        for k,v in tree.items():
            if isinstance(v,str) and v.startswith(symbols['global']):
                tree[k] = global_config[v.split(symbols['global'])[1]]
            elif isinstance(v,dict) or isinstance(v,list):
                recursive_global_replace(v,global_config)
    elif isinstance(tree,list):
        for k,v in enumerate(tree):
            if isinstance(v,str) and v.startswith(symbols['global']):
                tree[k] = global_config[v.split(symbols['global'])[1]]
            elif isinstance(v,dict) or isinstance(v,list):
                recursive_global_replace(v,global_config)

def set_config_parameters(config,parameters):
    config_to_edit = copy.deepcopy(config)
    recursive_global_replace(config_to_edit,parameters)
    
    return config_to_edit
