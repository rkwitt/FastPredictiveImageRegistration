require 'nn';
require 'nngraph';

create_model={}




function create_model.VAE_deformation_parallel_small_3D_noDropout(input_channel, patch_size, dim_hidden, feature_maps)
  filter_size = 3
  
  encoder = nn.Sequential()

  encoder:add(nn.VolumetricConvolution(input_channel, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  local layer_1 = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2, 1, 1, 1);  
  encoder:add(layer_1)


  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU()) 
  local layer_2 = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2, 1, 1, 1);  
  encoder:add(layer_2)    



  --decoder in x
  decoder = nn.Sequential()

  decoder:add(nn.VolumetricMaxUnpooling(layer_2))
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())

  decoder:add(nn.VolumetricMaxUnpooling(layer_1))
  decoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))

  --decoder in y
  decoder_2 = nn.Sequential()

  decoder_2:add(nn.VolumetricMaxUnpooling(layer_2))
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())

  decoder_2:add(nn.VolumetricMaxUnpooling(layer_1))
  decoder_2:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))

  decoder_3 = nn.Sequential()


  decoder_3:add(nn.VolumetricMaxUnpooling(layer_2))
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())

  decoder_3:add(nn.VolumetricMaxUnpooling(layer_1))
  decoder_3:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))


  local model = nn.Sequential()
  model:add(encoder)

  local reconstruct = nn.DepthConcat(2);
  reconstruct:add(decoder);
  reconstruct:add(decoder_2)
  reconstruct:add(decoder_3)
  model:add(reconstruct)

  collectgarbage()
  return model
end


function create_model.VAE_deformation_parallel_small_3D_traditionalDropout(input_channel, patch_size, dim_hidden, feature_maps)
  filter_size = 3
  
  encoder = nn.Sequential()

  encoder:add(nn.VolumetricConvolution(input_channel, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3, nil, true))
  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3, nil, true))
  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3, nil, true))
  local layer_1 = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2, 1, 1, 1);  
  encoder:add(layer_1)


  encoder:add(nn.VolumetricConvolution(feature_maps, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3, nil, true))
  encoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3, nil, true))
  encoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU()) 
  encoder:add(nn.Dropout(0.3, nil, true))
  local layer_2 = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2, 1, 1, 1);  
  encoder:add(layer_2)    


  --decoder in x
  decoder = nn.Sequential()


  decoder:add(nn.VolumetricMaxUnpooling(layer_2))
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3, nil, true))
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3, nil, true))
  decoder:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3, nil, true))

  decoder:add(nn.VolumetricMaxUnpooling(layer_1))
  decoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3, nil, true))
  decoder:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3, nil, true))
  decoder:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))

  --decoder in y
  decoder_2 = nn.Sequential()


  decoder_2:add(nn.VolumetricMaxUnpooling(layer_2))
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3, nil, true))
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3, nil, true))
  decoder_2:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3, nil, true))

  decoder_2:add(nn.VolumetricMaxUnpooling(layer_1))
  --decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder_2:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3, nil, true))
  decoder_2:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3, nil, true))
  decoder_2:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))

  decoder_3 = nn.Sequential()

  decoder_3:add(nn.VolumetricMaxUnpooling(layer_2))
  --decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.Dropout(0.3, nil, true))
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.Dropout(0.3, nil, true))
  decoder_3:add(nn.VolumetricConvolution(feature_maps*2, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.Dropout(0.3, nil, true))

  decoder_3:add(nn.VolumetricMaxUnpooling(layer_1))
  --decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder_3:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.Dropout(0.3, nil, true))
  decoder_3:add(nn.VolumetricConvolution(feature_maps, feature_maps, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_3:add(nn.PReLU())
  decoder_3:add(nn.Dropout(0.3, nil, true))
  decoder_3:add(nn.VolumetricConvolution(feature_maps, 1, filter_size, filter_size, filter_size, 1, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2), torch.floor(filter_size/2)))


  local model = nn.Sequential()
  model:add(encoder)

  local reconstruct = nn.DepthConcat(2);
  reconstruct:add(decoder);
  reconstruct:add(decoder_2)
  reconstruct:add(decoder_3)
  model:add(reconstruct)

  collectgarbage()
  return model
end


function create_model.VAE_deformation_parallel_small_noDropout(input_channel, patch_size, dim_hidden, feature_maps)
  filter_size = 3
  
  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(input_channel, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  local layer_1 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1);  
  encoder:add(layer_1)


  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())  
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU()) 
  local layer_2 = nn.SpatialMaxPooling(2, 2, 2, 2);  
  encoder:add(layer_2)    


  --decoder in x
  decoder = nn.Sequential()

  decoder:add(nn.SpatialMaxUnpooling(layer_2))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())

  decoder:add(nn.SpatialMaxUnpooling(layer_1))
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))

  --decoder in y
  decoder_2 = nn.Sequential()

  decoder_2:add(nn.SpatialMaxUnpooling(layer_2))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())

  decoder_2:add(nn.SpatialMaxUnpooling(layer_1))
  --decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())

  decoder_2:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))


  local model = nn.Sequential()
  model:add(encoder)

  local reconstruct = nn.DepthConcat(2);
  reconstruct:add(decoder);
  reconstruct:add(decoder_2)
  model:add(reconstruct)

  collectgarbage()
  return model
end

function create_model.VAE_deformation_parallel_small_fullSpatialDropout(input_channel, patch_size, dim_hidden, feature_maps)
  filter_size = 3
  
  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(input_channel, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialDropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialDropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialDropout(0.3))
  local layer_1 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1);  
  encoder:add(layer_1)


  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.SpatialDropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())  
  encoder:add(nn.SpatialDropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU()) 
  encoder:add(nn.SpatialDropout(0.3))
  local layer_2 = nn.SpatialMaxPooling(2, 2, 2, 2);  
  encoder:add(layer_2)    

  --decoder in x
  decoder = nn.Sequential()


  decoder:add(nn.SpatialMaxUnpooling(layer_2))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialDropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialDropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialDropout(0.3))

  decoder:add(nn.SpatialMaxUnpooling(layer_1))
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialDropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.SpatialDropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))

  --decoder in y
  decoder_2 = nn.Sequential()


  decoder_2:add(nn.SpatialMaxUnpooling(layer_2))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialDropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialDropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialDropout(0.3))

  decoder_2:add(nn.SpatialMaxUnpooling(layer_1))
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialDropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.SpatialDropout(0.3))

  decoder_2:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))


  local model = nn.Sequential()
  model:add(encoder)

  local reconstruct = nn.DepthConcat(2);
  reconstruct:add(decoder);
  reconstruct:add(decoder_2)
  model:add(reconstruct)

  collectgarbage()
  return model
end


function create_model.VAE_deformation_parallel_small_fullDropout(input_channel, patch_size, dim_hidden, feature_maps)
  filter_size = 3
  
  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(input_channel, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3))
  local layer_1 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1);  
  encoder:add(layer_1)


  encoder:add(nn.SpatialConvolution(feature_maps, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())
  encoder:add(nn.Dropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU())  
  encoder:add(nn.Dropout(0.3))
  encoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  encoder:add(nn.PReLU()) 
  encoder:add(nn.Dropout(0.3))
  local layer_2 = nn.SpatialMaxPooling(2, 2, 2, 2);  
  encoder:add(layer_2)    

  --decoder in x
  decoder = nn.Sequential()

  decoder:add(nn.SpatialMaxUnpooling(layer_2))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3))

  decoder:add(nn.SpatialMaxUnpooling(layer_1))
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder:add(nn.PReLU())
  decoder:add(nn.Dropout(0.3))
  decoder:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))

  --decoder in y
  decoder_2 = nn.Sequential()

  decoder_2:add(nn.SpatialMaxUnpooling(layer_2))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps*2, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps*2, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3))

  decoder_2:add(nn.SpatialMaxUnpooling(layer_1))
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3))
  decoder_2:add(nn.SpatialConvolution(feature_maps, feature_maps, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))
  decoder_2:add(nn.PReLU())
  decoder_2:add(nn.Dropout(0.3))

  decoder_2:add(nn.SpatialConvolution(feature_maps, 1, filter_size, filter_size, 1, 1, torch.floor(filter_size/2), torch.floor(filter_size/2)))


  local model = nn.Sequential()
  model:add(encoder)

  local reconstruct = nn.DepthConcat(2);
  reconstruct:add(decoder);
  reconstruct:add(decoder_2)
  model:add(reconstruct)

  collectgarbage()
  return model
end

return create_model