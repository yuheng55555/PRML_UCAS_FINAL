import wandb

def test_wandb():
	try:
		# 尝试初始化一个测试项目
		wandb.init(
			project="test-project",
			name="test-run",
			config={
				"learning_rate": 0.001,
				"batch_size": 32,
			}
		)
		print("Successfully initialized wandb!")
		
		# 尝试记录一些数据
		wandb.log({"test_metric": 0.5})
		print("Successfully logged data to wandb!")
		
		wandb.finish()
		return True
	except Exception as e:
		print(f"Error with wandb: {str(e)}")
		return False

if __name__ == "__main__":
	print("Testing wandb connection...")
	test_wandb()