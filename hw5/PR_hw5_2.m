% PR_hw5_2 Spectral Clustering£ºNg Algorithm
clear all;
close all;
clc;

X = load('data_hw5.txt');
plot(X(:,1), X(:,2), 'k.');hold on;
[m,n] = size(X);
% sigma = 35 is perfect!!!
sigma = 35;
% Knear = 20 is perfect!!!
Knear = 25;
class_num = 2;
chose_function = 2;

% 1.Generate Similarity Matrix W
W = zeros(m,m);
for i = 1:m
    for j = 1:m
        if i~=j
            W(i,j) = exp(-norm(X(i,:)-X(j,:))^2/(2*sigma^2));
        end
    end
end

% 2.Find the Knear for W
Wsort = zeros(m,m);
Windex = zeros(m,m);
for i = 1:m
	% reorder for W's per row
   [Wsort(i,:),Windex(i,:)] = sort(W(i,:),'descend');
end
for i = 1:m
    in_row_index = Windex(i,:);
    in_row_need_to_clear = in_row_index(Knear+1:m);
    W(i,in_row_need_to_clear) = 0;
end
% for symmetry
W = (W'+W)/2;

% 3.Generate normalized Laplace Matrix L_sym by calculating Degree Matrix D
D = zeros(m,m);
for i = 1:m
    D(i,i) = sum(W(i,:));
end
L = D-W;
L_sym = D^(-.5)*L*D^(-.5);

% 4.Calculate Knear eigenvectors & eigenvalues of L_sym as new feature U
lamda=zeros(m);
U = zeros(m,Knear);
[Vect,lamdaMat]=eig(L_sym);
for i = 1:m
    lamda(i)=lamdaMat(i,i);
end
[sortLamda,indexLamda]=sort(lamda);
countu = 0;
for k = 1:Knear
    countu = countu+1;
    U(:,countu)=Vect(:,indexLamda(k));  
end

% 5.Normalized U to T & Map X to Y by T
T = zeros(m,Knear);
U_col_sum = zeros(Knear);
Y = zeros(m,Knear);
for j = 1:Knear
    U_col_sum(j) = sqrt(sum(U(:,j))^2);
    for i = 1:m
        T(i,j) = U(i,j)/U_col_sum(j);
    end
end
Y = T;

% 6.K-means
% !!!ATTENTION ADJUST THE PARAMETER: sigma, Knear
% use 5.1 or 5.2 or 5.3 to finish the kmeans

if chose_function == 1
    %------------------ 5.1.use MATLAB function kmeans
    [class_type,Center] = kmeans(Y,class_num);
elseif chose_function == 2
    %------------------ 5.2.use my function K_means
    target = [linspace(1,1,100),linspace(2,2,100)];
    [ Center,class_type,iteration,acc,mse ]  = K_Means( Y,target,class_num );
else
    %------------------ 5.3.re-write k-means
    label = ones(1,m);
    pre_mu = zeros(class_num,Knear);
    mu = zeros(class_num,Knear);
    max_each_dim = max(Y,[],1);
    min_each_dim = min(Y,[],1);
    for i = 1:class_num
        mu(i,:) = min_each_dim+(max_each_dim-min_each_dim)*rand();
    end
    % real_mu = [0,0.01,0,0.01,0,0.01,0.01,0,0,0.01;
    %         0.01,0,0.01,0,0.01,0,0,0.01,0.01,0];

    while (sum(sum(abs(mu - pre_mu) > 1e-5)) > 0)
        pre_mu = mu;
        % give label for each sample by min_distance
        for i=1:m
            min_dis = Inf;
            belong_class = 1;
            for j=1:class_num
                cur_dis = sum((mu(j,:)-Y(i,:)).^2);
                if cur_dis < min_dis
                    min_dis = cur_dis;
                    belong_class = j;
                end
            end
            label(i) = belong_class;
        end
        % re-compute class center mu
        for k=1:class_num
            class_index = find(label==k);
            n = size(class_index,2);
            mu(k,:) = sum(Y(class_index,:))./n;
        end
    end
    class_type = label;
    Center = mu;
end

figure(2)
for i = 1:m
    if class_type(i)==1
        plot(X(i,1), X(i,2), 'r.'); hold on;
    else
        plot(X(i,1), X(i,2), 'b.'); hold on;
    end
end
style = ['r','b'];
% plot center
% for i = 1:class_num
%     plot(Center(i,1), Center(i,2), [style(i),'o']);
% end

right_num = 0;
for i = 1:m
    if (i<=100 && class_type(i)==1)||(i>100 && class_type(i)==2)
        right_num = right_num+1;
    end
end
accuracy = max(right_num/m,1-right_num/m);

