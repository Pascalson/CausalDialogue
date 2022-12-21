from graphviz import Digraph
import random
random.seed(123)

def int2chr(num:int):
    if num < 26:
        return chr(65+num)
    else:
        return chr(97+(num-26))

class DialNode:
    def __init__(self, node_data):
        self._type = node_data["type"]
        self._text = node_data["text"]
        if self._type == "utterance":
            self._speaker = node_data["speaker"]
        self._id = int(node_data["id"])
        self._prev_ids = node_data["prev_ids"]
        self._next_ids = []

    def __str__(self):
        returned_dict = {
            "type": self._type,
            "text": self._text,
        }
        if self._type == "utterance":
            returned_dict["speaker"] = self._speaker
        return str(returned_dict)

    def get_id(self):
        return self._id

    def reset_id(self, new_id):
        self._id = new_id

    def add_next_ids(self, next_ids:list):
        self._next_ids.extend(next_ids)

    def get_next_ids(self):
        return self._next_ids

    def reset_next_ids(self, new_next_ids):
        self._next_ids = new_next_ids

    def get_prev_ids(self):
        return self._prev_ids

    def add_prev_ids(self, prev_ids:list):
        self._prev_ids.extend(prev_ids)
        self._prev_ids = list(set(self._prev_ids))

    def reset_prev_ids(self, new_prev_ids):
        self._prev_ids = new_prev_ids

    def get_speaker(self):
        if self._type == "utterance":
            return self._speaker
        else:
            return "scene"

    def get_utt(self):
        return self._text

    def get_type(self):
        return self._type

        
class DialDAG:
    def __init__(self, dialogue):
        self.nodes = {}
        for utt in dialogue:
            self.nodes[int(utt["id"])] = DialNode(utt)
        for node in self.nodes.values():
            for prev_id in node.get_prev_ids():
                self.nodes[prev_id].add_next_ids([node.get_id()])

    def __str__(self):
        return str({k:str(v) for k,v in self.nodes.items()})

    def get_init_nodes(self):
        return [self.nodes[0]]

    def get_next_nodes(self, node_id):
        next_nodes = []
        for next_id in self.nodes[node_id].get_next_ids():
            if node_id == 0 and next_id == 0:
                continue
            next_nodes.append(self.nodes[next_id])
        if len(next_nodes) > 0:
            return next_nodes
        else:
            return None

    def add_utt_node(self, speaker, utt, prev_ids):
        node = {
            "id":max(self.nodes.keys())+1,
            "type":"utterance",
            "text":utt,
            "speaker":speaker,
            "prev_ids":prev_ids,
        }
        self.nodes[node["id"]] = DialNode(node)
        for prev_id in node["prev_ids"]:
            self.nodes[prev_id].add_next_ids([node["id"]])
            
    def plot(self, filepath=None):
        dot = Digraph()
        edges = []
        for node_id, node in self.nodes.items():
            if node.get_type() == "utterance":
                #dot.node(int2chr(node.get_id()), node.get_speaker()+": "+node.get_utt())
                dot.node(int2chr(node.get_id()), f"[{node.get_id()}] "+node.get_speaker()+": "+node.get_utt())
            else:
                #dot.node(int2chr(node.get_id()), node.get_utt())
                dot.node(int2chr(node.get_id()), f"[{node.get_id()}] "+node.get_utt())
            for prev_id in node.get_prev_ids():
                edges.append(int2chr(prev_id)+int2chr(node.get_id()))
        dot.edges(edges)
        if filepath == None:
            dot.render(view=True)
        else:
            dot.render(filename=filepath, cleanup=True)

    def dfs(self, node_id, visited, path=[], printed_paths=[]):
        path.append(node_id)
        visited[node_id] = True

        if len(self.nodes[node_id].get_next_ids()) == 0:
            #print(*path)
            printed_paths.append(list(path))

        for next_node_id in self.nodes[node_id].get_next_ids():
            if not visited[next_node_id]:
                self.dfs(next_node_id, visited, path, printed_paths)

        path.pop()
        visited[node_id] = False
    
    def get_possible_paths(self, expansion_rate=0.1):
        visited = {k:False for k in self.nodes.keys()}
        printed_paths = []
        self.dfs(0, visited, path=[], printed_paths=printed_paths)
        expanded_paths = []
        for path in printed_paths:
            if random.uniform(0,1) <= expansion_rate:
                expanded_paths.append(path)
        return printed_paths + expanded_paths
