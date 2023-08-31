classdef reshapeLayer < nnet.layer.Layer
    % Example custom PReLU layer.
    properties
        size
    end
    
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficient
        
    end
    
    methods
        function layer = reshapeLayer(size, name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % with numChannels channels and specifies the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "reshape with output size" + num2str(size);
            layer.size = size;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            if numel(layer.size) == 4
            %Z = reshape(X, layer.size(3), layer.size(4), layer.size(2), []);
            %dlarray(,"BSSC"); have to be unlabled dlarray! 
%             Z = reshape(X, [], 4, 4, 256);
%             Z = permute(Z, [2,3,4,1]);
            Z = reshape(X, layer.size(3), layer.size(4), layer.size(2), []);
            elseif numel(layer.size) == 3
            Z = reshape(X, layer.size(3), layer.size(2), []);
            end
        end
    end
end