# Following links are what we referred to for data generation
#https://github.com/huaxiuyao/HSML/blob/master/data_generator.py
#https://github.com/timchen0618/pytorch-leo/blob/master/data.py

import numpy as np
import os
import random
from PIL import Image
import pdb
import itertools
import pickle
import torch 
import torch.nn.functional as F


from collections import Counter
import gin
import tensorflow as tf
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline


class GaussianProcess(object):
    def __init__(self, batch_size, num_classes, data_source, is_train):
        self.batch_size = batch_size
        self.num_classes = num_classes 
        
    def generate_batch(self, is_test):
        dim_input = 1
        dim_output = 1
        batch_size = self.batch_size
        update_batch_size = np.random.randint(5, 11)
        update_batch_size_eval = 30 - update_batch_size
        num_samples_per_class = update_batch_size + update_batch_size_eval
        
        l = 0.5
        sigma = 1.0
        
        sel_set = np.zeros(batch_size)

        if is_test:
            init_inputs = np.zeros([batch_size, 1000, dim_input])
            outputs = np.zeros([batch_size, 1000, dim_output])
        else:
            init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
            outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            if is_test:
                init_inputs[func] = np.expand_dims(np.linspace(-2.0, 2.0, num=1000), axis=1)
            else:
                init_inputs[func] = np.random.uniform(-2.0, 2.0,
                                                      size=(num_samples_per_class, dim_input))
                
            x1 = np.expand_dims(init_inputs[func], axis=0)
            x2 = np.expand_dims(init_inputs[func], axis=1)
            
            kernel = sigma**2 * np.exp(-0.5 * np.square(x1-x2) / l**2)
            kernel = np.sum(kernel, axis=-1)
            kernel += (0.02 ** 2) * np.identity(init_inputs[func].shape[0])
            cholesky = np.linalg.cholesky(kernel)
            outputs[func] = np.matmul(cholesky, np.random.normal(size=(init_inputs[func].shape[0], dim_output)))
            
        rng = np.random.default_rng()
        random_idx = rng.permutation(init_inputs.shape[1])
        
        Cx = init_inputs[:, random_idx[:update_batch_size], :]
        Cy = outputs[:, random_idx[:update_batch_size], :]
        Tx = init_inputs
        Ty = outputs
        
        funcs_params = {'l': l, 'sigma': sigma}
        return (Cx, Tx), (Cy, Ty), funcs_params, sel_set


class mini_tiered_ImageNet(object):
    def __init__(self, batch_size, update_batch_size, update_batch_size_eval, num_classes, data_source, is_train):
        self.batch_size = batch_size
        self.update_batch_size = update_batch_size
        self.update_batch_size_eval = update_batch_size_eval
        self.num_classes = num_classes
        self.data_source = data_source
        self.is_train = is_train
        
        if is_train:
            self.metasplit = ['train', 'val']
        else:
            self.metasplit = ['train', 'val', 'test']
        
        self.construct_data()
    
    def construct_data(self):
        # loading embeddings
        self.embeddings = {}
        for d in self.metasplit:
            self.embeddings[d] = pickle.load(open(os.path.join('../data/'+self.data_source, d+'_embeddings.pkl'), 'rb'), encoding='latin1')
       
        # sort images by class
        self.image_by_class = {}
        self.embed_by_name = {}
        self.class_list = {}
        for d in self.metasplit:
            self.image_by_class[d] = {}
            self.embed_by_name[d] = {}
            self.class_list[d] = set()
            keys = self.embeddings[d]["keys"]
            for i, k in enumerate(keys):
                _, class_name, img_name = k.split('-')
                if (class_name not in self.image_by_class[d]):
                    self.image_by_class[d][class_name] = []
                self.image_by_class[d][class_name].append(img_name) 
                self.embed_by_name[d][img_name] = self.embeddings[d]["embeddings"][i]
                # construct class list
                self.class_list[d].add(class_name)
            
            self.class_list[d] = list(self.class_list[d])

    def generate_batch(self, is_test):
        # train_data -> [batch, N, k, dim]
        # valid_data -> [batch, N, k, dim]
        if self.is_train:
            if is_test:
                metasplit = 'val'
            else:
                metasplit = 'train'
        else:
            if is_test:
                metasplit = 'test'
            else:
                metasplit = np.random.choice(['train', 'val'])
        
        B = self.batch_size
        N = self.num_classes
        K = self.update_batch_size
        val_steps = self.update_batch_size_eval

        batch = {'input':[], 'label':[]}
        for b in range(B):
            shuffled_classes = self.class_list[metasplit].copy()
            random.shuffle(shuffled_classes)

            shuffled_classes = shuffled_classes[:N]

            inp = [[] for i in range(N)]
            lab = [[] for i in range(N)]

            for c, class_name in enumerate(shuffled_classes):
                images = np.random.choice(self.image_by_class[metasplit][class_name], K + val_steps)
                for i in range(K + val_steps):
                    embed = self.embed_by_name[metasplit][images[i]]
                    inp[c].append(embed)
                    lab[c].append(c)

            permutations = list(itertools.permutations(range(N)))
            order = random.choice(permutations)
            inputs = [inp[i] for i in order]
            labels = [lab[i] for i in order]

            batch['input'].append(np.asarray(inputs).reshape(N, K + val_steps, -1))
            batch['label'].append(np.asarray(labels).reshape(N, K + val_steps, -1))
            
        # convert to tensor
        input_tensor = torch.from_numpy(np.array(batch['input'])).permute(0,2,1,3)
        label_tensor = torch.from_numpy(np.array(batch['label'])).permute(0,2,1,3)
        
        image_batch = F.normalize(input_tensor, dim = -1)
        label_batch = torch.eye(N)[label_tensor].squeeze(-2)
        
        image_batch = image_batch.reshape(B, N * (K + val_steps), -1) 
        label_batch = label_batch.reshape(B, N * (K + val_steps), -1) 
        
        return image_batch, label_batch
            
        
class Heterogeneous_dataset(object):
    def __init__(self, batch_size, update_batch_size, update_batch_size_eval, num_classes, index, data_source, is_train):
        self.batch_size = batch_size
        self.update_batch_size = update_batch_size
        self.update_batch_size_eval = update_batch_size_eval
        self.num_samples_per_class = update_batch_size + update_batch_size_eval 
        self.num_classes = num_classes
        self.index = index
        
        if data_source == '1D':
            pass
        if data_source == 'multidataset':
            self.generate_multidataset_folder(is_train)
        
    def generate_1D_batch(self, is_test):
        dim_input = 1
        dim_output = 1
        batch_size = self.batch_size
        num_samples_per_class = self.num_samples_per_class

        # sine
        amp = np.random.uniform(0.1, 5.0, size=batch_size)
        phase = np.random.uniform(0., 2 * np.pi, size=batch_size)
        freq = np.random.uniform(0.8, 1.2, size=batch_size)

        # linear
        A = np.random.uniform(-3.0, 3.0, size=batch_size)
        b = np.random.uniform(-3.0, 3.0, size=batch_size)

        # quadratic
        A_q = np.random.uniform(-0.2, 0.2, size=batch_size)
        b_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        c_q = np.random.uniform(-3.0, 3.0, size=batch_size)

        # cubic
        A_c = np.random.uniform(-0.1, 0.1, size=batch_size)
        b_c = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_c = np.random.uniform(-2.0, 2.0, size=batch_size)
        d_c = np.random.uniform(-3.0, 3.0, size=batch_size)

        sel_set = np.zeros(batch_size)

        if is_test:
            init_inputs = np.zeros([batch_size, 1000, dim_input])
            outputs = np.zeros([batch_size, 1000, dim_output])
        else:
            init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
            outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            if is_test:
                init_inputs[func] = np.expand_dims(np.linspace(-5.0, 5.0, num=1000), axis=1)
            else:
                init_inputs[func] = np.random.uniform(-5.0, 5.0,
                                                      size=(num_samples_per_class, dim_input))
            #pretrain
            if self.index == None:
                sel = np.random.randint(4)
            
            #finetuning, data_loader0
            elif self.index == 0:
                sel = np.random.choice([0,1,2,3], p=[584/610, 9/610, 10/610, 7/610])
            
            #finetuning, data_loader1
            elif self.index == 1:
                sel = np.random.choice([0,1,2,3], p=[11/1890, 589/1890, 614/1890, 676/1890])
            
            if sel == 0:
                outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
            elif sel == 1:
                outputs[func] = A[func] * init_inputs[func] + b[func]
            elif sel == 2:
                outputs[func] = A_q[func] * np.square(init_inputs[func]) + b_q[func] * init_inputs[func] + c_q[func]
            elif sel == 3:
                outputs[func] = A_c[func] * np.power(init_inputs[func], np.tile([3], init_inputs[func].shape)) + b_c[
                    func] * np.square(init_inputs[func]) + c_c[func] * init_inputs[func] + d_c[func]
            
            sel_set[func] = sel
        funcs_params = {'amp': amp, 'phase': phase, 'freq': freq, 'A': A, 'b': b, 'A_q': A_q, 'c_q': c_q, 'b_q': b_q,
                        'A_c': A_c, 'b_c': b_c, 'c_c': c_c, 'd_c': d_c}
        return init_inputs, outputs, funcs_params, sel_set
    
    def generate_multidataset_folder(self, is_train):
        self.img_size = (84, 84)
        self.dim_input = np.prod(self.img_size) * 3
        dim_output = self.num_classes
        
        multidataset = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
        metatrain_folders, metatest_folders = [], []
        for eachdataset in multidataset:
            metatrain_folders.append(
                [os.path.join('{0}/multidataset/{1}/train'.format('../data', eachdataset), label) \
                 for label in os.listdir('{0}/multidataset/{1}/train'.format('../data', eachdataset)) \
                 if
                 os.path.isdir(os.path.join('{0}/multidataset/{1}/train'.format('../data', eachdataset), label)) \
                 ])
            if is_train:
                metatest_folders.append(
                    [os.path.join('{0}/multidataset/{1}/val'.format('../data', eachdataset), label) \
                     for label in os.listdir('{0}/multidataset/{1}/val'.format('../data', eachdataset)) \
                     if os.path.isdir(
                        os.path.join('{0}/multidataset/{1}/val'.format('../data', eachdataset), label)) \
                     ])
            else:
                metatest_folders.append(
                    [os.path.join('{0}/multidataset/{1}/test'.format('../data', eachdataset), label) \
                     for label in os.listdir('{0}/multidataset/{1}/test'.format('../data', eachdataset)) \
                     if os.path.isdir(
                        os.path.join('{0}/multidataset/{1}/test'.format('../data', eachdataset), label)) \
                     ])    
        
        self.metatrain_character_folders = metatrain_folders
        self.metatest_character_folders = metatest_folders
    
    def generate_multidataset_batch(self, is_test):
        if is_test:
            folders = self.metatest_character_folders
        else:
            folders = self.metatrain_character_folders
           
        sel_set = np.zeros(self.batch_size)
        
        cnt = 0
        for itr in range(self.batch_size):
            #pretrain
            if self.index == None:
                sel = np.random.randint(4)
            
            #finetuning, data_loader0
            elif self.index == 0:
                sel = np.random.choice([0,1,2,3], p=[9/959, 0, 950/959, 0])
                
            #finetuning, data_loader1
            elif self.index == 1:
                sel = np.random.choice([0,1,2,3], p=[930/1019, 17/1019, 45/1019, 27/1019])
            
            #finetuning, data_loader2
            elif self.index == 2:
                sel = np.random.choice([0,1,2,3], p=[11/944, 12/944, 0, 921/944])
            
            #finetuning, data_loader3
            elif self.index == 3:
                sel = np.random.choice([0,1,2,3], p=[6/1078, 1023/1078, 0, 49/1078])
            
            sel_set[itr] = sel
            folder = folders[sel]
            sampled_classes = random.sample(folder, self.num_classes)
            
            labels_and_images = self.get_images(sampled_classes, range(self.num_classes),
                                                self.update_batch_size, self.update_batch_size_eval)
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            
            tmp = 0
            for f in filenames:
                image = Image.open(f)
                image = np.expand_dims(image, axis=0) / 255.0
                if tmp == 0:
                    image_task = image
                    tmp += 1
                else:
                    image_task = np.concatenate((image_task, image), axis=0)
            image_task = np.expand_dims(image_task, axis=0)
            
            label_task = np.eye(self.num_classes)[labels]           
            label_task = np.expand_dims(label_task, axis=0)
        
            if cnt == 0:
                image_batch = image_task
                label_batch = label_task
                cnt += 1
            else:
                image_batch = np.concatenate((image_batch, image_task), axis=0)
                label_batch = np.concatenate((label_batch, label_task), axis=0)
            
        return image_batch, label_batch, sel_set       

    
    def get_images(self, paths, labels, support_set_size, query_set_size):
        sampler = lambda x: random.sample(x, support_set_size + query_set_size)
    
        tmp_list = []
        for i, path in zip(labels, paths):
            image_list = os.listdir(path)
        
            if '.ipynb_checkpoints' in image_list:
                image_list.remove('.ipynb_checkpoints')
        
            for image in sampler(image_list):
                tmp_list.append((i, os.path.join(path, image)))
    
        num_samples = support_set_size + query_set_size
        support_set, query_set = [], []
        for idx in range(len(paths)):
            support_set += tmp_list[num_samples*idx:num_samples*idx+support_set_size]
            query_set += tmp_list[num_samples*idx+support_set_size:num_samples*(idx+1)]
    
        images = support_set + query_set
        return images
    

class Meta_dataset(object):
    def __init__(self, batch_size, is_train, is_test):
        self.batch_size = batch_size
        self.is_train = is_train
        
        BASE_PATH = 'meta_dataset/record'
        GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
        gin.parse_config_file(GIN_FILE_PATH)
                
        if is_train:
            if is_test:
                SPLIT = learning_spec.Split.VALID
                ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower', 'mscoco']
                
            else:
                SPLIT = learning_spec.Split.TRAIN
                ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower']
                   
        else:
            ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower', 'mscoco', 'traffic_sign']       
            SPLIT = learning_epc.Split.TEST
            
        use_bilevel_ontology_list = [False]*len(ALL_DATASETS)
        use_dag_ontology_list = [False]*len(ALL_DATASETS)
        use_bilevel_ontology_list[5] = True
        use_dag_ontology_list[4] = True
        
        all_dataset_specs = []
        for dataset_name in ALL_DATASETS:
            dataset_records_path = os.path.join(BASE_PATH, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)
        
        self.to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        self.to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2))).float()

        variable_ways_shots = config.EpisodeDescriptionConfig(
            num_query=None, num_support=None, num_ways=None)
 
        self.dataset = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=all_dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            episode_descr_config=variable_ways_shots,
            split=SPLIT,
            image_size=84,
            shuffle_buffer_size=300)

    def generate_batch(self):       
        def data_loader(n_batches, dataset):
            for idx, (e, src) in enumerate(dataset):
                if idx == n_batches:
                    break
                yield (self.to_torch_imgs(e[0]), self.to_torch_labels(e[1]),
                       self.to_torch_imgs(e[3]), self.to_torch_labels(e[4]), src)
        
        sel_set = np.zeros(self.batch_size)
        Cx, Cy, Tx, Ty = [], [], [], []
        for i, batch in enumerate(data_loader(n_batches=self.batch_size, dataset=self.dataset)):
            data_support, labels_support, data_query, labels_query, sel = [x for x in batch]
            labels_support = torch.eye(labels_support.max()+1)[labels_support]
            labels_query = torch.eye(labels_query.max()+1)[labels_query]
            
            sel_set[i] = sel.numpy()
            Cx.append(data_support)
            Cy.append(labels_support)
            Tx.append(torch.cat((data_support, data_query), dim=0))
            Ty.append(torch.cat((labels_support, labels_query), dim=0))
            
        return Cx, Cy, Tx, Ty, sel_set