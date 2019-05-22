#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import math
import time
import logging
import mxnet as mx
import numpy as np
from operator_py import svm_metric
from config import cfg
from io_hybrid import check_dir, save_model


def _check_const_params(pd_before, pd_after):
    def _nd_equal(nd_array_1, nd_array_2):
        return np.array_equal(nd_array_1.asnumpy(),nd_array_2.asnumpy())

    if pd_before.keys() != pd_after.keys():
        logging.info("Param-names have changed")
        return 0 
    else:
        for key in pd_before:
            if not _nd_equal(pd_before[key]._reduce(), pd_after[key]._reduce()):
                logging.info("Param: {} has changed".format(key)) 
                return 0
        return 1
        

def _acc_evaluator_gluon(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = list() 
        for x in data:
            # outputs.append(net(x))
            outputs.append(mx.nd.softmax(net(x)))
        metric.update(label, outputs)
    return metric.get()


def _trainer_gluon(net, lr, lr_sceduler, wd, mtm, kv):
    optimizer = {
        'learning_rate':    lr,
        'wd':               wd,
        'momentum':         mtm,
        'lr_scheduler':     lr_sceduler,
    }
    return mx.gluon.Trainer(net.collect_params(), cfg.TRAIN.OPTIMIZER, optimizer, kvstore=kv) 


def _step_one_epoch(branch_idx, epoch, data_iters, nets, trainers, metrics, batch_size, ctx, start_time, log_interval=40, master_forward_only=False):
    '''
    '''
    assert len(data_iters)==len(nets)==len(trainers)==2, logging.error('Number of data_iters, nets, and trainers must be exactly 2')
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for i, batch in enumerate(data_iters[0]):
        btic = time.time()
        data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = list()
        Losses = list()
        with mx.autograd.record():
            for x, y in zip(data, label):
                if master_forward_only:
                    with mx.autograd.predict_mode():
                        _ = nets[0](x)
                    z = nets[1](_)
                else:
                    z = nets[1](nets[0](x))
                L = loss(z, y)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                Losses.append(L)
                outputs.append(mx.nd.softmax(z))
            for _ in Losses:
                _.backward()
        if not master_forward_only:
            trainers[0].step(batch.data[0].shape[0])    
        trainers[1].step(batch.data[0].shape[0])     # input batch_size
        metrics.update(label, outputs)
        if i!=0 and log_interval and not (i)%log_interval:
            name, acc = metrics.get()
            # batch end callback
            if cfg.TRAIN.LOG_LR and master_forward_only:
                logging.info('Epoch[{}] Batch [{}]\tLearning-rate: master={},\tbranch={}'.format(epoch, i, 0, trainers[1].learning_rate))
            elif cfg.TRAIN.LOG_LR: 
                logging.info('Epoch[{}] Batch [{}]\tLearning-rate: master={},\tbranch={}'.format(epoch, i, trainers[0].learning_rate, trainers[1].learning_rate))
            logging.info('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))

    # epoch end callback
    name, acc = metrics.get()
    for i in range(len(name)):
        logging.info('Epoch[{}] Branch-{} Training: {}={:.6f}'.format(epoch, branch_idx, name[i], acc[i]))
    logging.info('Epoch[{}] Time Cost: {:.6f}'.format(epoch, time.time()-start_time))
    name, val_acc = _acc_evaluator_gluon(lambda x: nets[1](nets[0](x)), data_iters[1], ctx)
    logging.info('Epoch[{}] Branch-{} Validation: {}={:.6f}'.format(epoch, branch_idx, name, val_acc))


def inst_lr_scheduler(num_samples, batch_size, kv, begin_epoch=0, base_lr=0.1, lr_factor=1, step_epochs=None, max_epochs=10, warmup_epochs=0, warmup_begin_lr=0, warmup_mode='linear', mode="STEP_DECAY"):
    '''
    '''
    if mode == 'CONSTANT':  # constant learning rate
        logging.info('Using constant learning rate.')
        return base_lr, None
    elif mode == 'STEP_DECAY':
        logging.info('Using step_dacay mode to update learning rate.')
        assert num_samples and batch_size, logging.error('Invalid number of samples or mini-batch size per gpu')
        epoch_size = int(math.ceil(num_samples/batch_size))     # on all gpus
        if 'dist' in kv.type:       # distributed job
            epoch_size /= kv.num_workers
        warmup_steps = epoch_size * warmup_epochs 
        lr = base_lr
        for s in step_epochs:
            if begin_epoch >= s:
                lr *= lr_factor 
        if lr != base_lr:
            logging.info('Adjust learning rate to %e for epoch %d'%(lr, begin_epoch))
        steps = [epoch_size*(x-begin_epoch) for x in step_epochs if x-begin_epoch>0]
        if float('.'.join(mx.__version__.split('.')[:2])) < 1.5:    # lower version 
            logging.warning('Warmup is not supported on mxnet version lower than 1.5.0!')
            return lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor)
        else:
            return lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor, base_lr=base_lr, warmup_steps=warmup_steps, warmup_begin_lr=warmup_begin_lr, warmup_mode=warmup_mode)
    elif mode == 'COSINE_DECAY':    # mxnet_version >= 1.4.0
        epoch_size = int(math.ceil(num_samples/batch_size))     # on all gpus
        if 'dist' in kv.type:       # distributed job
            epoch_size /= kv.num_workers
        logging.info('Using cosine_dacay mode to update learning rate.')
        max_update = epoch_size * max_epochs
        warmup_steps = epoch_size * warmup_epochs 
        lr = base_lr
        return lr, mx.lr_scheduler.CosineScheduler(max_update, base_lr=base_lr, final_lr=0, warmup_steps=warmup_steps, warmup_begin_lr=warmup_begin_lr, warmup_mode=warmup_mode)


def inst_eval_metrics(lst_metrics, top_k=5):
    '''
    '''
    all_metrics = {
        'acc':          mx.metric.Accuracy(),
        'ce':           mx.metric.CrossEntropy(),
        'f1':           mx.metric.F1(),
        'mae':          mx.metric.MAE(),
        'mse':          mx.metric.MSE(),
        'rmse':         mx.metric.RMSE(),
        'top_k_acc':    mx.metric.TopKAccuracy(top_k=top_k),
        'hl':           svm_metric.HingeLoss()
    }
    eval_metrics = mx.metric.CompositeEvalMetric()
    for metric in lst_metrics:
        assert metric in all_metrics, logging.error('Invalid evaluation metric!')
        eval_metrics.add(all_metrics[metric])
        logging.info('{} added into evaluation metric list'.format(metric))
    return eval_metrics


def get_batch_end_callback(batch_size, display_lr=True, display_batch=20):
    '''
    '''
    def _cbs_metrics(cb_eval_metrics):
        logging.debug(type(cb_eval_metrics))
        metric, value = cb_eval_metrics.get()
        str_metric = str()
        for index, met in enumerate(metric):
            str_metric += '\t{}={}'.format(met, value[index])
        return str_metric

    def _display_metrics(BatchEndParam):
        '''
        call back eval metrics info
        '''
        if BatchEndParam.nbatch % display_batch == 0:
            logging.info("Epoch[{}] Batch [{}]\t{}".format(BatchEndParam.epoch, BatchEndParam.nbatch, _cbs_metrics(BatchEndParam.eval_metric)))

    def _display_lr(BatchEndParam):
        '''
        call back learning rate info
        '''
        if BatchEndParam.nbatch != 0 and BatchEndParam.nbatch % display_batch == 0:
            logging.info("Epoch[{}] Batch [{}]\tlearning-rate={}".format(
                BatchEndParam.epoch, BatchEndParam.nbatch, BatchEndParam.locals["self"]._optimizer._get_lr(0)))

    cbs = list()
    if display_lr: 
        cbs.append(_display_lr)
    cbs.append(mx.callback.Speedometer(batch_size, display_batch))
    return cbs


def alternate_train_gluon(phase, nets, train_iters, dev_iters, num_samples, lrs, lr_schedulers, batch_size, metrics, ctx, epochs=10, weight_decays=list(), momentums=list(), log_interval=40):
    '''
    '''
    assert phase in [1,2], logging.error('Value of phase should be 1 or 2 for now')
    assert len(weight_decays)==len(momentums)==len(nets), logging.error('Invalid number of weight decays or momentums')

    # kv = mx.kvstore.create(cfg.TRAIN.KV_STORE)
    kv = cfg.TRAIN.KV_STORE
    model_prefix = cfg.TRAIN.OUTPUT_MODEL_PREFIX
    check_dir(model_prefix)
    
    # evaluate before training 
    if cfg.TRAIN.PRE_EVALUATION:
        logging.info("Evaluate sub-nets on validation-data before training:")
        for i in range(1, len(nets)):
            name, val_acc = _acc_evaluator_gluon(lambda x: nets[i](nets[0](x)), dev_iters[i-1], ctx)
            logging.info('Branch-{} Validation: {}={:.6f}'.format(i, name, val_acc))

    if phase == 1:
        # instantiate trainers
        trainer_master = _trainer_gluon(nets[0], lrs[0], lr_schedulers[0], weight_decays[0], momentums[0], kv) 
        trainer_branch_1 = _trainer_gluon(nets[1], lrs[1], lr_schedulers[1], weight_decays[1], momentums[1], kv)
        trainer_branch_2 = _trainer_gluon(nets[2], lrs[2], lr_schedulers[2], weight_decays[2], momentums[2], kv)

        for epoch in range(epochs):
            # Train branch 1 on dataset 1
            tic = time.time()
            train_iters[0].reset()
            metrics.reset()
            _step_one_epoch(1, epoch, (train_iters[0], dev_iters[0]), (nets[0], nets[1]), (trainer_master, trainer_branch_1), metrics, batch_size, ctx, tic, log_interval=log_interval, master_forward_only=False)
            
            # Train branch 2 on dataset 2
            tic = time.time()
            train_iters[1].reset()
            metrics.reset()
            _step_one_epoch(2, epoch, (train_iters[1], dev_iters[1]), (nets[0], nets[2]), (trainer_master, trainer_branch_2), metrics, batch_size, ctx, tic, log_interval=log_interval, master_forward_only=False)

            # save models
            if (epoch+1) % cfg.TRAIN.SAVE_INTERVAL == 0:
                nets[0].collect_params().save('{}-master-phase-{}-{:0>4}.params'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, phase, (epoch+1)))
                nets[0](mx.sym.Variable('data')).save('{}-master-phase-{}-symbol.json'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, phase))
                for i in range(len(nets)-1):
                    nets[i+1].collect_params().save('{}-branch-{}-phase-{}-{:0>4}.params'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, i+1, phase, (epoch+1)))
                    nets[i+1](mx.sym.Variable('data')).save('{}-branch-{}-phase-{}-symbol.json'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, i+1, phase))

    elif phase == 2:
        # instantiate trainers
        trainer_master = "dummy trainer" 
        trainer_branch_1 = _trainer_gluon(nets[1], lrs[1], lr_schedulers[1], weight_decays[1], momentums[1], kv)
        trainer_branch_2 = _trainer_gluon(nets[2], lrs[2], lr_schedulers[2], weight_decays[2], momentums[2], kv)

        master_before = nets[0].collect_params()
        for epoch in range(epochs):
            # Train branch 1 on dataset 1
            tic = time.time()
            train_iters[0].reset()
            metrics.reset()
            _step_one_epoch(1, epoch, (train_iters[0], dev_iters[0]), (nets[0], nets[1]), (trainer_master, trainer_branch_1), metrics, batch_size, ctx, tic, log_interval=log_interval, master_forward_only=True)
            
            # Train branch 2 on dataset 2
            tic = time.time()
            train_iters[1].reset()
            metrics.reset()
            _step_one_epoch(2, epoch, (train_iters[1], dev_iters[1]), (nets[0], nets[2]), (trainer_master, trainer_branch_2), metrics, batch_size, ctx, tic, log_interval=log_interval, master_forward_only=True)

            # save models
            if (epoch+1) % cfg.TRAIN.SAVE_INTERVAL == 0:
                nets[0].collect_params().save('{}-master-phase-{}-{:0>4}.params'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, phase, (epoch+1)))
                nets[0](mx.sym.Variable('data')).save('{}-master-phase-{}-symbol.json'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, phase))
                for i in range(len(nets)-1):
                    nets[i+1].collect_params().save('{}-branch-{}-phase-{}-{:0>4}.params'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, i+1, phase, (epoch+1)))
                    nets[i+1](mx.sym.Variable('data')).save('{}-branch-{}-phase-{}-symbol.json'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, i+1, phase))
        
        # check if params of master net have changed
        master_after = nets[0].collect_params()
        check = _check_const_params(master_before, master_after)
        if check:
            logging.info("Master-net has successfully passed params checking")
        else:
            logging.info("Params of master-net have changed, which should not happen in phase-2")

    else:
        pass


def generic_train(train_iter, dev_iter, symbol, arg_params, aux_params, num_samples, batch_size, begin_epoch):
    '''
    '''
    # initialization
    kv = mx.kvstore.create(cfg.TRAIN.KV_STORE)
    checkpoint = save_model(cfg.TRAIN.OUTPUT_MODEL_PREFIX, kv.rank)
    devices = [mx.gpu(x) for x in cfg.TRAIN.GPU_IDX] if cfg.TRAIN.USE_GPU else mx.cpu() 
    label_names = ['softmax_label'] if cfg.TRAIN.USE_SOFTMAX else ['svm_label']
    lr, lr_scheduler = inst_lr_scheduler(num_samples, batch_size, kv, begin_epoch=begin_epoch, base_lr=cfg.TRAIN.BASE_LR, lr_factor=cfg.TRAIN.LR_FACTOR, step_epochs=cfg.TRAIN.STEP_EPOCHS, max_epochs=cfg.TRAIN.MAX_EPOCHS, warmup_epochs=cfg.TRAIN.WARMUP_EPOCHS, warmup_begin_lr=cfg.TRAIN.WARMUP_BEGIN_LR, warmup_mode=cfg.TRAIN.WARMUP_MODE, mode=cfg.TRAIN.LR_DECAY_MODE) 
    optimizer_params = {'learning_rate': lr,
                        'momentum': cfg.TRAIN.MOMENTUM,
                        'wd': cfg.TRAIN.WEIGHT_DECAY, 
                        'lr_scheduler': lr_scheduler}
    batch_end_callbacks = get_batch_end_callback(batch_size, display_lr=cfg.TRAIN.LOG_LR, display_batch=cfg.TRAIN.LOG_INTERVAL) 
    if "top_k_acc" in cfg.TRAIN.METRICS:
        metrics = inst_eval_metrics(cfg.TRAIN.METRICS, top_k=cfg.TRAIN.METRICS_TOP_K_ACC) 
    else:
        metrics = inst_eval_metrics(cfg.TRAIN.METRICS) 

    # bind module 
    if cfg.TRAIN.FT.FREEZE_WEIGHTS: 
        # fix all weights except for fc layers
        freeze_list = [k for k in arg_params if 'fc' not in k] 
        mod = mx.mod.Module(symbol=symbol, context=devices, label_names=label_names, fixed_param_names=freeze_list)
    else:
        mod = mx.mod.Module(symbol=symbol, context=devices, label_names=label_names)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # init weights
    if cfg.TRAIN.XAVIER_INIT:
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    else:
        mod.init_params(initializer=mx.init.Normal())

    # set weights 
    if cfg.TRAIN.FINETUNE:
        # replace all parameters except for the last fully-connected layer with pre-trained model
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    elif cfg.TRAIN.RESUME:
        mod.set_params(arg_params, aux_params, allow_missing=False, allow_extra=False)
    else:
        pass
    
    # fit 
    mod.fit(train_iter, 
            dev_iter,
            eval_metric=metrics,
            begin_epoch=begin_epoch,
            num_epoch=cfg.TRAIN.MAX_EPOCHS,
            kvstore=kv,
            optimizer=cfg.TRAIN.OPTIMIZER,
            optimizer_params=optimizer_params,
            batch_end_callback=batch_end_callbacks,
            epoch_end_callback=checkpoint,
            allow_missing=True)

    return mod.score(dev_iter, metrics)
