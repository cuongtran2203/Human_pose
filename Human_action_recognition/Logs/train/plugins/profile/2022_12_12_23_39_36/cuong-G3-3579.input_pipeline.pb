  *	
?Zd?Z@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate{??B???!-??x?CJ@)=b??BW??1%Q??H@:Preprocessing2U
Iterator::Model::ParallelMapV2F\ ?K??!????,@)F\ ?K??1????,@:Preprocessing2F
Iterator::ModelaP???b??!`??KU?9@)|DL?$z??1??̐R'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?5?;Nс?!o?l?O @)?5?;Nс?1o?l?O @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE??f?R??!vU????-@)Ý#??}?1td?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDOʤ?6??!?????R@)@?z??{u?1??A??@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[1]::FromTensor?p??[um?!6nw$??
@)?p??[um?16nw$??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~P)???!?<S??K@)G仔?d\?1qT?T???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[0]::TensorSlice??hUMP?!?C?????)??hUMP?1?C?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.