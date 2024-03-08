#########################################################
####### Simulate synthetic data from learned tree #######
#########################################################
library(DirichletReg)
library(ggplot2)
library(dplyr)

N <- 5000 # number of objects
V <- 7 # number of vertices
U <- 6 # number of stages 

# true parameter on causal tree
causal_tree <- matrix(0,nrow=2,ncol=U)
causal_tree[,1] <- c(0.8,0.2)
causal_tree[,2] <- c(0.7,0.3)
causal_tree[,3] <- c(0.5,0.5)
causal_tree[,4] <- c(0.6,0.4)
causal_tree[,5] <- c(0.2,0.8)
causal_tree[,6] <- c(0.8,0.2)
# true parameter on learn tree
learn_tree <- matrix(0,nrow=2,ncol=U)
learn_tree[,1] <- c(0.544,0.456)
learn_tree[,2] <- c(12/17,5/17)
learn_tree[,3] <- c(52/57,5/57)
learn_tree[,4] <- c(0.875,0.125)
learn_tree[,5] <- c(3/26,23/26)
learn_tree[,6] <- c(0.5,0.5)

# simulate data
# tree paths of the N objects:
tree <- matrix(0,nrow=N,ncol=3)
tree[,1] <- rbinom(5000,1,learn_tree[2,1]) + 1
tree[tree[,1] == 1,2] <- rbinom(sum(tree[,1] == 1),1,learn_tree[2,2]) + 3
tree[tree[,1] == 2,2] <- rbinom(sum(tree[,1] == 2),1,learn_tree[2,3]) + 5
tree[tree[,1] == 1 & tree[,2] == 3,3] <- rbinom(sum(tree[,1] == 1 & tree[,2] == 3),1,learn_tree[2,4])+7
tree[tree[,1] == 1 & tree[,2] == 4,3] <- rbinom(sum(tree[,1] == 1 & tree[,2] == 4),1,learn_tree[2,6])+9
tree[tree[,1] == 2 & tree[,2] == 5,3] <- rbinom(sum(tree[,1] == 2 & tree[,2] == 5),1,learn_tree[2,5])+11
tree[tree[,1] == 2 & tree[,2] == 6,3] <- rbinom(sum(tree[,1] == 2 & tree[,2] == 6),1,learn_tree[2,6])+13

N_f <- sum(tree[,1] == 1) # number of failures (2732)
# assume endogenous failure may miss root cause
# the probability of missing (non-perfect remedy) is
p_m <- 0.3 # p(delta=0|endogenous,failure,rootcause1)
           #=p(delta=0|endogenous,failure,rootcause2)
           #=p(delta=0|endogenous,failure) = 0.3
# potentially missing paths are lambda = [1,3, *]
poss_miss_tree <- tree[tree[,1] == 1 & tree[,2] == 3,]
delta <- rbinom(nrow(poss_miss_tree),1,1-p_m) # delta=0 meaning missing root cause [imperfect/uncertain remedy]
# we masking these missing values when estimating


##########################################
##### MCMC for parameter estimation ######
##########################################

alpha <- 5/c(8,8)
mcmc_parest <- function(delta,poss_miss_tree,alpha,iter,eps){
  perfect <- poss_miss_tree[delta==1,]
  uncertain <- poss_miss_tree[delta==0,]
  N_rc1_perfect <- sum(perfect[,3] == 7)
  N_rc2_perfect <- sum(perfect[,3] == 8)
  N_uncertain <- dim(uncertain)[1]
 
  theta_1 <- runif(1,0,1)
  theta_2 <- 1 - theta_1
  theta_old <- c(theta_1,theta_2)
  
  
  #return out
  out <- matrix(0,nrow=2,ncol=iter)
  lgpost <- rep(0,iter)
  for(i in 1:iter){
    theta_1 <- runif(1,max(0,theta_old[1]-eps),min(1,theta_old[1]+eps))
    theta_new <- c(theta_1,1-theta_1)
    prop_old <- log(1/(min(1,theta_new[1]+eps)-max(0,theta_new[1]-eps)))
    prop_new <- log(1/(min(1,theta_old[1]+eps)- max(0,theta_old[1]-eps)))
    target_old <- ddirichlet(matrix(theta_old,ncol=2),alpha,log = TRUE,sum.up = TRUE) + 
      N_rc1_perfect * log(theta_old[1]) + N_rc2_perfect * log(theta_old[2]) +
      N_uncertain * log(theta_old[1]^2 + theta_old[2]^2)
    target_new <- ddirichlet(matrix(theta_new,ncol=2),alpha,log = TRUE,sum.up = TRUE) + 
      N_rc1_perfect * log(theta_new[1]) + N_rc2_perfect * log(theta_new[2]) +
      N_uncertain * log(theta_new[1]^2 + theta_new[2]^2)

    accept <- min(0,target_new + prop_old - target_old - prop_new)
    
    if(log(runif(1,0,1)) < accept){
      out[,i] <- theta_new 
      theta_old <- theta_new
      lgpost[i] <- target_new
    }else{
      out[,i] <- theta_old 
      lgpost[i] <- target_old
    }
  }
  return(list(out=out,lgpost=lgpost))
}

posterior <- mcmc_parest(delta,poss_miss_tree,alpha,iter=5000,eps=0.05) 
lgpost <- posterior$lgpost
posterior_est <- posterior$out
plot(posterior_est[1,],type='l')

# remove burn-in (first 500)
plot(posterior_est[1,1001:5000],type='l')
hist(posterior_est[1,1001:5000])
plot(posterior_est[2,1001:5000],type='l')
hist(posterior_est[2,1001:5000])
posterior_mean <- apply(posterior_est[,1001:5000],1,mean)
posterior_mean
# bias

true_mean <- learn_tree[,4]
posterior_mean - true_mean

# situation error
sit_error_1 <- sum((posterior_mean - true_mean)^2)
sit_error_1

# to merge floret:
potential_merge <- tree[tree[,1] == 2 & tree[,2] == 5,]
N_potential_merge <- c(sum(potential_merge[,3] == 11),sum(potential_merge[,3] == 12))
post_alpha <- alpha + N_potential_merge
posterior_mean_potential <- post_alpha/sum(post_alpha)

true_mean_2 <- learn_tree[,5]
sit_error_2 <- sum((posterior_mean_potential - true_mean_2)^2)
sit_error_2

sit_error_sum <- sit_error_1 + sit_error_2

lgp_separate <- ddirichlet(matrix(posterior_mean_potential,ncol=2),alpha,log=TRUE,sum.up = TRUE)+
 mean(lgpost[1001:5000])
  


##############################################
### structural learning -- model selection ###
##############################################
### if they are merged to a stage:

alpha_stage <- 10/c(8,8)
mcmc_stageest <- function(delta,poss_miss_tree,alpha_stage,N_potential_merge,iter,eps){
  perfect <- poss_miss_tree[delta==1,]
  uncertain <- poss_miss_tree[delta==0,]
  N_rc1_perfect <- length(perfect[,3] == 7)
  N_rc2_perfect <- length(perfect[,3] == 8)
  N_uncertain <- dim(uncertain)[1]
  
  theta_1 <- runif(1,0,1)
  theta_2 <- 1 - theta_1
  theta_old <- c(theta_1,theta_2)
  
  #return out
  out <- matrix(0,nrow=2,ncol=iter)
  lgpost <- rep(0,iter)
  for(i in 1:iter){
    theta_1 <- runif(1,max(0,theta_old[1]-eps),min(1,theta_old[1]+eps))
    theta_new <- c(theta_1,1-theta_1)
    prop_old <- log(1/(min(1,theta_new[1]+eps)-max(0,theta_new[1]-eps)))
    prop_new <- log(1/(min(1,theta_old[1]+eps)- max(0,theta_old[1]-eps)))
    
    target_old <- ddirichlet(matrix(theta_old,ncol=2),alpha_stage,log = TRUE,sum.up = TRUE) + 
      (N_potential_merge[1] + N_rc1_perfect) * log(theta_old[1]) + (N_potential_merge[2] + N_rc2_perfect) * log(theta_old[2]) +
      N_uncertain * log(theta_old[1]^2 + theta_old[2]^2)
    target_new <- ddirichlet(matrix(theta_new,ncol=2),alpha_stage,log = TRUE,sum.up = TRUE) + 
      (N_potential_merge[1] + N_rc1_perfect) * log(theta_new[1]) + (N_potential_merge[2] + N_rc2_perfect) * log(theta_new[2]) +
      N_uncertain * log(theta_new[1]^2 + theta_new[2]^2)
    
    accept <- min(0,target_new + prop_old - target_old - prop_new)
    
    if(log(runif(1,0,1)) < accept){
      out[,i] <- theta_new 
      theta_old <- theta_new
      lgpost[i] <- target_new
    }else{
      out[,i] <- theta_old 
      lgpost[i] <- target_old
    }
  }
  return(list(out=out,lgpost=lgpost))
}

posterior_stage <- mcmc_stageest(delta,poss_miss_tree,alpha_stage,N_potential_merge,iter=5000,eps=0.05) 
lgpost_stage <- posterior_stage$lgpost
posterior_stageest <- posterior_stage$out
plot(posterior_stageest[1,],type='l')
# remove burn-in (first 1000)
plot(posterior_stageest[1,1001:5000],type='l')
hist(posterior_stageest[1,1001:5000])
plot(posterior_stageest[2,1001:5000],type='l')
hist(posterior_stageest[2,1001:5000])
posterior_stagemean <- apply(posterior_stageest[,1001:5000],1,mean)
posterior_stagemean
# situation error
stage_error_1 <- sum((posterior_stagemean - true_mean)^2)
stage_error_1
stage_error_2 <- sum((posterior_stagemean - true_mean_2)^2)
stage_error_2
stage_error_sum <- stage_error_1 + stage_error_2
stage_error_sum
sit_error_sum
# sit_error_sum < stage_error_sum, suggesting not merge

lgp_together <- mean(lgpost_stage[1001:5000])
lgp_together
lgp_separate
# lgp_together <  lgp_separate, suggesting not merge


# the score of the rest of the tree
lgscore <- function(prior,new){
  new_sum <- sum(new)
  prior_sum <- sum(prior)
  return(lgamma(prior_sum) - lgamma(new_sum) - sum(lgamma(prior) - lgamma(new)))
}

# posterior parameters excluding the uncertain floret
newpar <- matrix(0,nrow = 2,ncol = 6)
alpha0 <- 5
newpar[,1] <- c(alpha0/2,alpha0/2) + c(sum(tree[,1] == 1),sum(tree[,1] == 2))
newpar[,2] <- c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 3),sum(tree[,2] == 4))
newpar[,3] <- c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 5),sum(tree[,2] == 6))
newpar[,4] <- c(alpha0/8,alpha0/8) + c(sum(tree[,3] == 11),sum(tree[,3] == 12))
newpar[,5] <- c(alpha0/4,alpha0/4) + c(sum(tree[,3] == 9) + sum(tree[,3]==13),sum(tree[,3] == 10)+sum(tree[,3] == 14))

prior <- newpar
prior[,1] <- c(alpha0/2,alpha0/2)
prior[,2] <- c(alpha0/4,alpha0/4)
prior[,3] <- c(alpha0/4,alpha0/4)
prior[,4] <- c(alpha0/8,alpha0/8)
prior[,5] <- c(alpha0/4,alpha0/4)

score <- 0
for(i in 1:5){
  score <- score + lgscore(prior[,i],newpar[,i])
}

lgp_together + score
lgp_separate + score

#####################################
### situational error figure ########
#####################################

posterior_tree <- matrix(0,nrow=2,ncol=6)
alpha0 <- 5
posterior_tree[,1] <-(c(alpha0/2,alpha0/2) + c(sum(tree[,1] == 1),sum(tree[,1] == 2)))/sum(c(alpha0/2,alpha0/2) + c(sum(tree[,1] == 1),sum(tree[,1] == 2)))
posterior_tree[,2] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 3),sum(tree[,2] == 4)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 3),sum(tree[,2] == 4)))
posterior_tree[,3] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 5),sum(tree[,2] == 6)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 5),sum(tree[,2] == 6)))
posterior_tree[,4] <- c(posterior_mean[1],1-posterior_mean[1])
posterior_tree[,5] <-(c(alpha0/8,alpha0/8) + c(sum(tree[,3] == 11),sum(tree[,3] == 12)))/sum(c(alpha0/8,alpha0/8) + c(sum(tree[,3] == 11),sum(tree[,3] == 12)))
posterior_tree[,6] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,3] == 9) + sum(tree[,3]==13),sum(tree[,3] == 10)+sum(tree[,3] == 14)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,3] == 9) + sum(tree[,3]==13),sum(tree[,3] == 10)+sum(tree[,3] == 14)))



sit_error_tree <- rep(0,6)
for(i in 1:6){
  sit_error_tree[i] <- sum((posterior_tree[,i] - learn_tree[,i])^2)
}



###############################
### intervention adjustment ###
###############################
posterior_est_pre <- posterior_est[,1001:5000]
## shift back to causal tree
p_endo <- posterior_tree[1,1] * posterior_tree[1,2] + 
  posterior_tree[2,1] * posterior_tree[1,3]
p_exo <- 1 - p_endo
p_nf_endo_rc1 <- posterior_tree[2,1] * posterior_tree[1,3] * posterior_tree[1,5]
p_f_endo_rc1 <- posterior_tree[1,1] * posterior_tree[1,2] * posterior_est_pre[1,]
p_f_endo_rc2 <- posterior_tree[1,1] * posterior_tree[1,2] * (1-posterior_est_pre[1,])

p_rc1.en <- (p_f_endo_rc1 + p_nf_endo_rc1)/p_endo
post_p_rc1.en <- p_rc1.en


perfect <- poss_miss_tree[delta==1,]
uncertain <- poss_miss_tree[delta==0,]
N_rc1_perfect <- sum(perfect[,3] == 7)
N_rc2_perfect <- sum(perfect[,3] == 8)
N_uncertain <- dim(uncertain)[1]

discount <- 0.8
for(i in 1:length(post_p_rc1.en)){
  pre <- p_rc1.en[i]
  Nr1 <- pre * N_uncertain + N_rc1_perfect
  Nr2 <- (1-pre) * N_uncertain + N_rc2_perfect
  ratio <- (pre/(1-pre))*discount^{(Nr1-Nr2)/N}
  post_p_rc1.en[i] <- ratio/(1+ratio)
}

df <- data.frame(x = c(rep('pre',length(post_p_rc1.en)),
                       rep('post',length(post_p_rc1.en))),
                 y = c(p_rc1.en,post_p_rc1.en))
df.mu <- data.frame(x=c('pre','post'),
                    mean=c(mean(p_rc1.en),mean(post_p_rc1.en)))

ggplot(df,aes(x=y,color=x))+geom_density(aes(color=x),size=0.9)+
  geom_vline(data=df.mu, aes(xintercept=mean, color=x),
             linetype="dashed")+
  labs(x='p(root cause 1|endogenous cause)',y='density',color='')+
  scale_color_manual(name = "intervention", values = c(1,4))+
  theme_bw(base_size = 12)+
  theme(legend.position = c(0.85,0.85),legend.background=element_rect(),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 13),
        axis.title=element_text(size=15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))




p1.post <- p_f_endo_rc1 * post_p_rc1.en/p_rc1.en
p2.post <- p_f_endo_rc2 * (1-post_p_rc1.en)/(1-p_rc1.en)

p1 <- c()
p2 <- c()
for(i in 1:4000){
  p1 <- c(p1, posterior_tree[1,1] * posterior_tree[1,2] * posterior_est[1,1000+i])
  p2 <- c(p2, posterior_tree[1,1] * posterior_tree[1,2] * posterior_est[2,1000+i])
}
pred <- data.frame(p = c(p1,p2,p1.post,p2.post),
                   rc = c(rep(c("1","2"),each = length(p1)),
                          rep(c("1","2"),each = length(p1))),
                   intervene = c(rep("pre",length(p1)*2),
                                 rep("post",length(p1)*2)))
pred.mu <- data.frame(rc=c("1","2","1","2"),
                    mean=c(mean(p1),mean(p2),mean(p1.post),mean(p2.post)),
                    intervene = c("pre","pre","post","post"))


ggplot(pred,aes(x=p,color=interaction(rc,intervene)))+
         geom_density()+
  geom_vline(data=pred.mu, aes(xintercept=mean, color=interaction(rc,intervene)),
             linetype="dashed")+
  labs(x='p(endogenous failure)',y='density',color='')+
  scale_color_manual(name = "root cause.intervention", values = c(1,4,7,8))+
  theme_bw(base_size = 12)+
  theme(legend.position = c(0.8,0.7),legend.background=element_rect(),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        axis.title=element_text(size=15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))+
  scale_x_continuous(limits=c(0,1),breaks = seq(0,1,0.1))



learn_to_causal <- function(learnest=posterior_tree){
  # input the parameter estimates for the learned tree
  causalest <- learnest
  p_endo <- learnest[1,1] * learnest[1,2] + learnest[2,1] * learnest[1,3]
  p_exo <- 1 - p_endo
  p_rc1.endo <- (learnest[1,1] * learnest[1,2] * learnest[1,4] + 
                   learnest[2,1] * learnest[1,3] * learnest[1,5])/p_endo
  p_rc2.endo <- 1 - p_rc1.endo
  p_c1.exo <- (learnest[1,1] * learnest[2,2] * learnest[1,6] +
                 learnest[2,1] * learnest[2,3] * learnest[1,6])/p_exo
  p_c2.exo <- 1 - p_c1.exo
  p_f.endo.rc1 <- learnest[1,1] * learnest[1,2] * learnest[1,4]/(p_endo * p_rc1.endo)
  p_nf.endo.rc1 <- 1-p_f.endo.rc1
  p_f.endo.rc2 <- learnest[1,1] * learnest[1,2] * learnest[2,4]/(p_endo * p_rc2.endo)
  p_nf.endo.rc2 <- 1-p_f.endo.rc2
  p_f.exo.c1 <- learnest[1,1] * learnest[2,2] * learnest[1,6]/(p_exo*p_c1.exo)
  p_nf.exo.c1 <- 1-p_f.exo.c1
  causalest[,1] <- c(p_endo,p_exo)
  causalest[,2] <- c(p_rc1.endo,p_rc2.endo)
  causalest[,3] <- c(p_c1.exo,p_c2.exo)
  causalest[,4] <- c(p_f.endo.rc1,p_nf.endo.rc1)
  causalest[,5] <- c(p_f.endo.rc2,p_nf.endo.rc2)
  causalest[,6] <- c(p_f.exo.c1,p_nf.exo.c1)
  return(causalest)
}
###############################
#### sensitivity analysis #####
###############################
hold_error <- c()
for(alpha0 in c(0.1,0.5,1,3,5,7,10)){
  posterior <- mcmc_parest(delta,poss_miss_tree,alpha=alpha/c(8,8),iter=5000,eps=0.05) 
  posterior_mean <- apply(posterior$out[,1001:5000],1,mean)
  posterior_tree <- matrix(0,nrow=2,ncol=6)
  posterior_tree[,1] <-(c(alpha0/2,alpha0/2) + c(sum(tree[,1] == 1),sum(tree[,1] == 2)))/sum(c(alpha0/2,alpha0/2) + c(sum(tree[,1] == 1),sum(tree[,1] == 2)))
  posterior_tree[,2] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 3),sum(tree[,2] == 4)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 3),sum(tree[,2] == 4)))
  posterior_tree[,3] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 5),sum(tree[,2] == 6)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,2] == 5),sum(tree[,2] == 6)))
  posterior_tree[,4] <- c(posterior_mean[1],1-posterior_mean[1])
  posterior_tree[,5] <-(c(alpha0/8,alpha0/8) + c(sum(tree[,3] == 11),sum(tree[,3] == 12)))/sum(c(alpha0/8,alpha0/8) + c(sum(tree[,3] == 11),sum(tree[,3] == 12)))
  posterior_tree[,6] <-(c(alpha0/4,alpha0/4) + c(sum(tree[,3] == 9) + sum(tree[,3]==13),sum(tree[,3] == 10)+sum(tree[,3] == 14)))/sum(c(alpha0/4,alpha0/4) + c(sum(tree[,3] == 9) + sum(tree[,3]==13),sum(tree[,3] == 10)+sum(tree[,3] == 14)))
  
  causalest <- learn_to_causal(posterior_tree)
  for(i in 1:6){
    hold_error <- c(hold_error,sum((causalest[,i] - causal_tree[,i])^2))
  }
}

situational <- data.frame(alpha0 = rep(c("0.1","0.5","1","3","5",'7','10'),each=6),error = hold_error,
                          stage = rep(1:6,7))
total_error <- as.data.frame(aggregate(situational$error, by=list(alpha0=situational$alpha0), FUN=sum))
total_error <- arrange(transform(total_error,
                                 alpha0=factor(alpha0,
                                               levels=c("0.1","0.5","1","3","5",'7','10'))),
                       alpha0)

ggplot(data = total_error,aes(x=alpha0,y=x,group=1))+
  geom_line()+geom_point() + 
  labs(x='alpha0',y='total situational error')+
  theme_bw(base_size = 12)+ 
  theme(axis.title=element_text(size=15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

situational <- arrange(transform(situational,
                              alpha0=factor(alpha0,
                                              levels=c("0.1","0.5","1","3","5",'7','10'))),
                    alpha0)
p0 <-ggplot(data = situational, aes(x=stage,y=error,group=alpha0))+
  geom_line(aes(color=alpha0),size=0.9)+
  labs(x='Stage',y='Situational error',color='') + 
  scale_color_manual(name = "alpha0", values = c(1,2,3,4,5,6,7))+
  theme_bw(base_size = 12)+ 
  theme(legend.position = c(0.8,0.65),legend.background=element_rect(),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        axis.title=element_text(size=15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))+ 
  scale_x_discrete(limits=c("1","2","3","4","5","6"))
p0
