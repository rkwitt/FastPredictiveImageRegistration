require 'nn';
require 'nngraph';
require 'torch';
require 'image';

local data_provider={}

function data_provider.create(filename)
	local dataset = torch.load(filename)
	-- local max_v = 1500 --for OASIS data
	--local max_v = dataset.images:max() -- for BRATs data
	--local temp1 = dataset.images:ge(max_v):float():mul(max_v)
	--local temp2 = dataset.images:lt(max_v):float():cmul(dataset.images)
	--dataset.images = temp1 + temp2
	--dataset.images = dataset.images:mul(1/max_v)
	result_dataset = {}
	result_dataset.images = dataset:float()
	result_dataset.n_images = dataset:size(1)
	return result_dataset 
end

function data_provider.get_image(dataset, image_idx)
	return dataset.images[image_idx]
end

return data_provider
