import modal
import modal.experimental


app = modal.App()


vol = modal.Volume.from_name("RLHF")
@app.function(volumes={"/data": vol})                                                                                          
def run():
    with open("/data/xyz.txt", "w") as f:
        f.write("hello")
    vol.commit()



@app.local_entrypoint()
def main(): 
    run.remote()

