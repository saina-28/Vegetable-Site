const cartModel = require("../models/Cart")

const checkCart = async (req, res, next) => {
    try{
        // const userid = req.cookies.userid;
        const userid = req.userid
        const findCart = await cartModel.findOne({ userid : userid })
        if(!findCart)
        {
            const result = await cartModel.create({
                userid : userid,
                cartItems : []
            })
            console.log("User Cart Created Successfully")
        }
        next();
    }
    catch(error)
    {
        console.log("User not Logged in! -> Error in checkCart");
        next()
    }
}
module.exports = {checkCart};