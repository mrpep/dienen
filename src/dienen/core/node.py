class DienenNode():
    """
    Base class for Dienen classes, to set configuration files and validate args/kwargs
    """
    def __init__(self, name):
    	self.valid_nodes = []
    	self.required_nodes = []

    def _check_valid_nodes(self,config):
        #Check if the user entered a wrong node or typos:
        for key in list(config.keys()):
            if key not in self.valid_nodes:
                raise Exception('{} is not recognized as a Dienen parameter'.format(key))
        #Check if required nodes are set:
        for node in self.required_nodes:
        	if node not in list(config.keys()):
        		raise Exception('{} is needed to train the model'.format(node))

    def set_config(self,config):
        self.config = config
        self._check_valid_nodes(self.config)

    def set_predict_config(self,config):
        self.predict_config = config
        self._check_valid_nodes(self.predict_config)
