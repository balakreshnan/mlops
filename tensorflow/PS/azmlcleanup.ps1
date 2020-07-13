az login

az extension add -n azure-cli-ml

az ml folder attach -w gputraining -g mlops

az ml run list --experiment-name "tf-mnist"

az ml run list --experiment-name "tf-mnist" --query 'target' -o json

az ml run list --experiment-name "tf-mnist" --query [].[target,run_id,experiment_id] --output table

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[target,run_id,experiment_id,Status]" --output table


az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table

az ml run cancel -r runid -w workspace_name -e experiment_name


pools=$(az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]")
for pool in $pools
{
    echo $pool
}

while [ $(az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]"
do    
    echo -n "."
    sleep 5   
done
echo "Preview is registered"

az ml run list --query '[?contains(target, "sdk")]' --output table

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table | \
while read g n; \
do \
  echo "jj"; \
done

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table | while read line

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[run_id]" --output table >> names.txt

az ml run cancel -r runid -w workspace_name -e experiment_name. 


foreach($line in Get-Content names.txt) {
  az ml run cancel --run $line
}


az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[run_id]" --output table

6760c046-a768-4661-8db1-48d880e7192b
4ad29037-81e6-4e36-a775-62c011393acc
09eb5a23-6f63-4c13-9c9f-a8741f3f2579
5486b4d4-11c7-4293-88b7-dab8f9aeb00a
efc37927-97e1-45f7-8db3-605d4110ed09
ae07f101-8fe4-47ad-98e0-c48fa33211e3
1962a68b-5d57-4e6d-8141-6eefac82494c
410f62c2-9acd-4611-83e5-2c2158aa3b2a
2431085a-345d-4074-a111-f680c31ddf71
8739f7a2-1246-41bd-8cdb-de520c99d356
5473fa41-e5f1-4ed2-a102-2611cd1dc121
708f7dbf-7ac8-419d-9a9d-eab3af19bee5
f9335e5b-7e75-4ef3-a532-a49cd63bd3de
32c72a66-21c2-40e6-bd30-e30136d6c089
411d9af6-2f89-4a6c-b667-21db631e2a96
317871fa-20b5-4266-b394-af7e98a675c3
abca392c-4e43-435d-9e9e-3396dd3dbe34
29755a71-ebe2-4528-8aef-1d2d00dd5b4a

az ml folder attach -w gputraining -g mlops

az ml run cancel -r 29755a71-ebe2-4528-8aef-1d2d00dd5b4a -w gputraining -e tf-mnist 

az ml run cancel -r 6760c046-a768-4661-8db1-48d880e7192b -w gputraining -e tf-mnist
az ml run cancel -r 4ad29037-81e6-4e36-a775-62c011393acc -w gputraining -e tf-mnist
az ml run cancel -r 09eb5a23-6f63-4c13-9c9f-a8741f3f2579 -w gputraining -e tf-mnist
az ml run cancel -r 5486b4d4-11c7-4293-88b7-dab8f9aeb00a -w gputraining -e tf-mnist
az ml run cancel -r efc37927-97e1-45f7-8db3-605d4110ed09 -w gputraining -e tf-mnist
az ml run cancel -r ae07f101-8fe4-47ad-98e0-c48fa33211e3 -w gputraining -e tf-mnist
az ml run cancel -r 1962a68b-5d57-4e6d-8141-6eefac82494c -w gputraining -e tf-mnist
az ml run cancel -r 410f62c2-9acd-4611-83e5-2c2158aa3b2a -w gputraining -e tf-mnist
az ml run cancel -r 2431085a-345d-4074-a111-f680c31ddf71 -w gputraining -e tf-mnist
az ml run cancel -r 8739f7a2-1246-41bd-8cdb-de520c99d356 -w gputraining -e tf-mnist
az ml run cancel -r 5473fa41-e5f1-4ed2-a102-2611cd1dc121 -w gputraining -e tf-mnist
az ml run cancel -r 708f7dbf-7ac8-419d-9a9d-eab3af19bee5 -w gputraining -e tf-mnist
az ml run cancel -r f9335e5b-7e75-4ef3-a532-a49cd63bd3de -w gputraining -e tf-mnist
az ml run cancel -r 32c72a66-21c2-40e6-bd30-e30136d6c089 -w gputraining -e tf-mnist
az ml run cancel -r 411d9af6-2f89-4a6c-b667-21db631e2a96 -w gputraining -e tf-mnist
az ml run cancel -r 317871fa-20b5-4266-b394-af7e98a675c3 -w gputraining -e tf-mnist
az ml run cancel -r abca392c-4e43-435d-9e9e-3396dd3dbe34 -w gputraining -e tf-mnist


7abe08bb-f51d-4107-aba2-c34c83d3c36d
6760c046-a768-4661-8db1-48d880e7192b

Run_number    Root_run_id                           Experiment_id                         Created_utc                  User_id                      
         Run_id                                Status    Start_time_utc               Heartbeat_enabled    Name                                     
 Data_container_id                          Hidden    Root_run_uuid                         Run_uuid                              Target    _experim
ent_name    End_time_utc


az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[run_id,status]" --output table

az ml run list --experiment-name "tf-mnist" --query "[?contains(status, 'Running')].[run_id,status]" --output table

az ml run cancel -r 7abe08bb-f51d-4107-aba2-c34c83d3c36d -w gputraining -e tf-mnist
az ml run cancel -r 6760c046-a768-4661-8db1-48d880e7192b -w gputraining -e tf-mnist