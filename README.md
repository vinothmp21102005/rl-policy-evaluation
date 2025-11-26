# POLICY EVALUATION

## AIM
Deploy the frozen-lake MDP. Find value function for both the policies given using policy evaluation and compare them.

## PROBLEM STATEMENT
This is an experiment in Reinforcement Learning where you compare different policies in a Frozen-Lake environment using policy evaluation

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
  prev_V=np.zeros(len(P))
  while True:
    V=np.zeros(len(P))
    for s in range(len(P)):
      for prob, next_state, reward, done in P[s][pi(s)]:
        V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))

    if np.max(np.abs(prev_V-V))<theta:
      break
    prev_V=V.copy()
  return V
```

## OUTPUT:
## POLICY 1:
<img width="1002" height="621" alt="image" src="https://github.com/user-attachments/assets/347bff45-da91-411a-903b-65c21c337f55" />

## POLICY 2:
<img width="872" height="470" alt="image" src="https://github.com/user-attachments/assets/6496413c-7f34-4213-a602-7996306f215b" />

<img width="517" height="102" alt="image" src="https://github.com/user-attachments/assets/248263c1-fe22-4fb1-a733-bf6daac61d52" />

## POLICY EVALUATION OF POLICIES:

<img width="628" height="425" alt="image" src="https://github.com/user-attachments/assets/8d249bd3-dd8b-4517-8460-d259cebd58c8" />

## COMPARING POLICIES:
<img width="995" height="472" alt="image" src="https://github.com/user-attachments/assets/6fb84115-75e3-4db2-ba47-08b5646a174b" />

## RESULT:

Therefore, policies are compared successfully using policy evaluation function in Frozen-Lake MDP.
