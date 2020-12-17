function [ reorder_mu,label,iteration,acc,mse ] = K_Means( data,target,k )
% K_Means function
% Input:
%  data: the points set with size(1000*2)
%  k: given the number of clusters, this time k=5
% Output:
%  reorder_mu: estimated means with size(5*2)
%  label: estimated laber for cluster
%  iteration: the times of iretation
%  acc: the accuracy of estimation
%  acc: the accuracy of estimation
    [m,n] = size(data);
    
    % 1.initialize means mu randomly
    max_each_dim = max(data,[],1);
    min_each_dim = min(data,[],1);
    mu = zeros(k,n);
    for i = 1:k
        mu(i,:) = min_each_dim+(max_each_dim-min_each_dim)*rand();
    end
    mu
    % ----
%     mu = [0,0.01,0,0.01,0,0.01,0.01,0,0,0.01;
%         0.01,0,0.01,0,0.01,0,0,0.01,0.01,0];
    % ----
    
    % 2.Loop to find constance means mu
    iteration = 0;
    while 1
        iteration = iteration + 1;
        pre_mu = mu;
        % 2.1.initialize temp{i} = x(i) - mu(j) for computing distance, temp's size (k*m)
        for i = 1:k
            temp1{i} = [];
            for j = 1:m
                temp1{i} = [temp1{i};data(j,:) - mu(i,:)];
            end
        end
        % 2.2.compute distance with size (m*k) like one-hot model
        distance = zeros(m,k);
        for i = 1:m
            temp2 = [];
            for j = 1:k
                temp2 = [temp2 norm(temp1{j}(i,:))];
            end
            [~, idx] = min(temp2);
            distance(i,idx) = 1;  
        end
        % 2.3.re-compute mu
        for i = 1:k
            for j = 1:n
                mu(i,j) = sum(distance(:,i).*data(:,j))/sum(distance(:,i));
            end
        end
        % 2.4.close cycle condition
        % if norm(pre_mu-mu)<0.5
        if iteration == 1000
            break;
        end
    end
    mu
    
    % 3.compute real_mu & adjust mus' order
    real_mu = zeros(k,n);
    for i = 1:k
        temp_data_i = zeros(m,n);
        count = 0;
        for j = 1:m
            if target(j)==i
                temp_data_i(j,:) = data(j,:);
                count = count+1;
            end
        end
        real_mu(i,:) = sum(temp_data_i)/count;
    end
    real_mu
    reorder_mu = zeros(k,n);
    indexs = zeros(1,k);
    all_distance = zeros(1,k);
    for i = 1:k
        temp_dist = [];
        for j = 1:k
            temp_dist = [temp_dist norm(real_mu(i,:)-mu(j,:))];
        end
        temp_dist
        [val, idx] = min(temp_dist);
        indexs(i) = idx;
        all_distance(i) = val;
    end
    mse = mean(all_distance);
    all_distance
    mse
    indexs
    for i = 1:k
        reorder_mu(i,:) = mu(indexs(i),:);
    end
    reorder_mu
    style = ['r','b','k','g','m'];
    for i = 1:k
        plot(mu(i,1), mu(i,2), [style(i),'+']);hold on;
        plot(reorder_mu(i,1), reorder_mu(i,2), [style(i),'o']);
    end
    
    % 4.compute label & accuracy
    label = [];
    for i = 1:m
        temp3 = [];
        for j = 1:k
            temp3 = [temp3 norm(data(i,:)-reorder_mu(j,:))];
        end
        [~, index] = min(temp3);
        label = [label index];
    end
    correct_count = 0;
    for i = 1:m
        if label(i)==target(i)
            correct_count = correct_count+1;
        end
    end
    acc = correct_count/m;
end

